use nix::cmsg_space;
use nix::errno::Errno;
use nix::fcntl::{fcntl, FcntlArg, FdFlag};
use nix::sys::socket::{
    recvmsg, sendmsg, ControlMessage, ControlMessageOwned, MsgFlags,
};
use nix::unistd::{close, read, write};
use serde::{Deserialize, Serialize};
use std::io::{IoSlice, IoSliceMut};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use thiserror::Error;

pub const MAX_FRAME_BYTES: usize = 1_048_576;
const MAX_RIGHTS_FDS: usize = 16;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u8,
    pub operation: String,
    pub request_id: String,
    pub profile_sha256: String,
    pub payload: serde_json::Value,
    pub fd_roles: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub schema_version: u8,
    pub request_id: String,
    pub status: String,
    pub payload: serde_json::Value,
    pub error: Option<String>,
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("frame exceeds configured maximum")]
    FrameTooLarge,
    #[error("unexpected end of frame")]
    UnexpectedEof,
    #[error("ancillary data was truncated or malformed")]
    InvalidAncillary,
    #[error("fd roles do not exactly match the operation contract")]
    InvalidFdRoles,
    #[error("received descriptor is not close-on-exec")]
    DescriptorNotCloexec,
    #[error("non-canonical JSON frame")]
    NonCanonicalJson,
    #[error("JSON contract error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("system call failed: {0}")]
    System(#[from] Errno),
}

fn read_exact_fd(fd: RawFd, mut buffer: &mut [u8]) -> Result<(), ProtocolError> {
    while !buffer.is_empty() {
        match read(fd, buffer) {
            Ok(0) => return Err(ProtocolError::UnexpectedEof),
            Ok(count) => buffer = &mut buffer[count..],
            Err(Errno::EINTR) => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn write_all_fd(fd: RawFd, mut payload: &[u8]) -> Result<(), ProtocolError> {
    while !payload.is_empty() {
        match write(fd, payload) {
            Ok(0) => return Err(ProtocolError::UnexpectedEof),
            Ok(count) => payload = &payload[count..],
            Err(Errno::EINTR) => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn decode_length(header: [u8; 4], maximum: usize) -> Result<usize, ProtocolError> {
    let length = u32::from_be_bytes(header) as usize;
    if length > maximum || length > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    Ok(length)
}

fn decode_canonical_request(payload: &[u8]) -> Result<Request, ProtocolError> {
    let value: serde_json::Value = serde_json::from_slice(payload)?;
    if serde_json::to_vec(&value)? != payload {
        return Err(ProtocolError::NonCanonicalJson);
    }
    Ok(serde_json::from_value(value)?)
}

pub fn recv_request(
    socket: RawFd,
    expected_roles: &[&str],
) -> Result<(Request, Vec<OwnedFd>), ProtocolError> {
    let mut header = [0_u8; 4];
    let mut iov = [IoSliceMut::new(&mut header)];
    let mut ancillary = cmsg_space!([RawFd; MAX_RIGHTS_FDS]);
    let message = recvmsg::<()>(
        socket,
        &mut iov,
        Some(&mut ancillary),
        MsgFlags::MSG_CMSG_CLOEXEC | MsgFlags::MSG_WAITALL,
    )?;
    if message.bytes != 4
        || message.flags.intersects(MsgFlags::MSG_TRUNC | MsgFlags::MSG_CTRUNC)
    {
        return Err(ProtocolError::InvalidAncillary);
    }
    let mut raw_fds = Vec::new();
    for control in message.cmsgs() {
        match control {
            ControlMessageOwned::ScmRights(values) => raw_fds.extend(values),
            _ => {
                for fd in raw_fds {
                    let _ = close(fd);
                }
                return Err(ProtocolError::InvalidAncillary);
            }
        }
    }
    let owned_fds: Vec<OwnedFd> = raw_fds
        .into_iter()
        .map(|fd| unsafe { OwnedFd::from_raw_fd(fd) })
        .collect();
    let length = decode_length(header, MAX_FRAME_BYTES)?;
    let mut payload = vec![0_u8; length];
    read_exact_fd(socket, &mut payload)?;
    let request = decode_canonical_request(&payload)?;
    let roles_match = request.fd_roles.len() == expected_roles.len()
        && request
            .fd_roles
            .iter()
            .zip(expected_roles)
            .all(|(actual, expected)| actual == expected);
    if !roles_match || owned_fds.len() != expected_roles.len() {
        return Err(ProtocolError::InvalidFdRoles);
    }
    for fd in &owned_fds {
        let flags =
            FdFlag::from_bits_truncate(fcntl(fd.as_raw_fd(), FcntlArg::F_GETFD)?);
        if !flags.contains(FdFlag::FD_CLOEXEC) {
            return Err(ProtocolError::DescriptorNotCloexec);
        }
    }
    Ok((request, owned_fds))
}

pub fn send_response(
    socket: RawFd,
    response: &Response,
    handoff: Option<RawFd>,
) -> Result<(), ProtocolError> {
    let value = serde_json::to_value(response)?;
    let payload = serde_json::to_vec(&value)?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    let mut frame = Vec::with_capacity(payload.len() + 4);
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    if let Some(fd) = handoff {
        let rights = [fd];
        let sent = sendmsg::<()>(
            socket,
            &[IoSlice::new(&frame)],
            &[ControlMessage::ScmRights(&rights)],
            MsgFlags::empty(),
            None,
        )?;
        write_all_fd(socket, &frame[sent..])
    } else {
        write_all_fd(socket, &frame)
    }
}

pub fn recv_exact_frame(fd: RawFd, maximum: usize) -> Result<Vec<u8>, ProtocolError> {
    let mut header = [0_u8; 4];
    read_exact_fd(fd, &mut header)?;
    let length = decode_length(header, maximum)?;
    let mut payload = vec![0_u8; length];
    read_exact_fd(fd, &mut payload)?;
    Ok(payload)
}

pub fn send_exact_frame(fd: RawFd, payload: &[u8]) -> Result<(), ProtocolError> {
    if payload.len() > MAX_FRAME_BYTES {
        return Err(ProtocolError::FrameTooLarge);
    }
    let length = (payload.len() as u32).to_be_bytes();
    write_all_fd(fd, &length)?;
    write_all_fd(fd, payload)
}

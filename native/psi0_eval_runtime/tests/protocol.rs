use nix::sys::socket::{sendmsg, ControlMessage, MsgFlags};
use nix::unistd::{close, pipe};
use psi0_eval_runtime::protocol::{
    recv_exact_frame, recv_request, send_exact_frame, MAX_FRAME_BYTES,
};
use std::io::IoSlice;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::thread;

#[test]
fn exact_frame_round_trip() {
    let (sender, receiver) = UnixStream::pair().unwrap();
    let writer = thread::spawn(move || {
        send_exact_frame(sender.as_raw_fd(), b"canonical-frame").unwrap();
    });
    assert_eq!(
        recv_exact_frame(receiver.as_raw_fd(), MAX_FRAME_BYTES).unwrap(),
        b"canonical-frame"
    );
    writer.join().unwrap();
}

#[test]
fn oversized_frame_is_rejected_before_write() {
    let (sender, _receiver) = UnixStream::pair().unwrap();
    let payload = vec![0_u8; MAX_FRAME_BYTES + 1];
    assert!(send_exact_frame(sender.as_raw_fd(), &payload).is_err());
}

#[test]
fn request_receives_one_cloexec_fd_in_exact_role_order() {
    let (sender, receiver) = UnixStream::pair().unwrap();
    let (read_fd, write_fd) = pipe().unwrap();
    let payload = br#"{"fd_roles":["construction_lock"],"operation":"self_test","payload":{},"profile_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","request_id":"request-1","schema_version":1}"#;
    let mut frame = Vec::with_capacity(payload.len() + 4);
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(payload);
    let rights = [read_fd];
    sendmsg::<()>(
        sender.as_raw_fd(),
        &[IoSlice::new(&frame)],
        &[ControlMessage::ScmRights(&rights)],
        MsgFlags::empty(),
        None,
    )
    .unwrap();

    let (request, received) =
        recv_request(receiver.as_raw_fd(), &["construction_lock"]).unwrap();
    assert_eq!(request.request_id, "request-1");
    assert_eq!(received.len(), 1);
    close(read_fd).unwrap();
    close(write_fd).unwrap();
}

#[test]
fn request_rejects_unknown_keys_and_role_drift() {
    let (sender, receiver) = UnixStream::pair().unwrap();
    let payload = br#"{"extra":true,"fd_roles":[],"operation":"self_test","payload":{},"profile_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","request_id":"request-2","schema_version":1}"#;
    send_exact_frame(sender.as_raw_fd(), payload).unwrap();
    assert!(recv_request(receiver.as_raw_fd(), &[]).is_err());

    let (sender, receiver) = UnixStream::pair().unwrap();
    let payload = br#"{"fd_roles":[],"operation":"self_test","payload":{},"profile_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","request_id":"request-3","schema_version":1}"#;
    send_exact_frame(sender.as_raw_fd(), payload).unwrap();
    assert!(recv_request(receiver.as_raw_fd(), &["policy_connection"]).is_err());
}

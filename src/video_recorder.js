export default function ({ parentElement, setStateValue }) {
  const video = parentElement.querySelector('video');
  const start = parentElement.querySelector('#start');
  const stop = parentElement.querySelector('#stop');
  const clear = parentElement.querySelector('#clear');
  const status = parentElement.querySelector('#status');
  let stream, recorder, timer, url, disposed = false, cancelled = false;
  let chunks = [], bytes = 0, elapsed = 0;
  const limit = 20 * 1024 * 1024;
  function release() {
    clearInterval(timer);
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
    stop.disabled = true;
  }
  function revoke() { if (url) URL.revokeObjectURL(url); url = null; }
  function finish() {
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    clearInterval(timer);
    stop.disabled = true;
  }
  clear.onclick = () => {
    cancelled = true;
    finish(); release(); revoke();
    video.removeAttribute('src'); video.load();
    setStateValue('clip', null);
    start.disabled = false;
    status.textContent = 'ลบวิดีโอแล้ว';
  };
  stop.onclick = finish;
  start.onclick = async () => {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      status.textContent = 'เบราว์เซอร์นี้อัดวิดีโอไม่ได้ กรุณาเปิด localhost ใน Chrome หรือ Edge'; return;
    }
    start.disabled = true;
    clear.disabled = true;
    cancelled = false;
    status.textContent = 'กำลังขออนุญาตใช้กล้อง...';
    try {
      const acquired = await navigator.mediaDevices.getUserMedia({video: {width: {ideal:640}, height: {ideal:480}, frameRate: {ideal:15, max:30}}, audio:false});
      if (disposed) { acquired.getTracks().forEach(t => t.stop()); return; }
      stream = acquired;
      revoke();
      video.removeAttribute('src'); video.srcObject = stream; video.controls = false;
      await video.play();
      const mime = ['video/webm;codecs=vp8','video/webm','video/mp4'].find(t => MediaRecorder.isTypeSupported(t));
      if (!mime) throw new Error('ไม่พบรูปแบบวิดีโอที่รองรับ กรุณาใช้ Chrome หรือ Edge');
      chunks = []; bytes = 0; elapsed = 0;
      recorder = new MediaRecorder(stream, {mimeType:mime, videoBitsPerSecond:1500000});
      recorder.ondataavailable = e => {
        if (e.data.size) { chunks.push(e.data); bytes += e.data.size; }
        if (bytes > limit) { cancelled = true; status.textContent = 'วิดีโอใหญ่เกิน 20 MB กรุณาอัดใหม่ให้สั้นลง'; finish(); }
      };
      recorder.onerror = () => { cancelled = true; status.textContent = 'บันทึกไม่สำเร็จ กรุณาลองใหม่'; finish(); release(); start.disabled = false; };
      recorder.onstop = async () => {
        release();
        if (disposed || cancelled) { start.disabled = false; return; }
        const blob = new Blob(chunks, {type:recorder.mimeType || mime});
        if (!blob.size || blob.size > limit) { status.textContent = 'วิดีโอว่างหรือใหญ่เกิน 20 MB'; start.disabled = false; return; }
        status.textContent = 'กำลังส่งวิดีโอ...';
        const reader = new FileReader();
        reader.onload = () => {
          if (disposed || cancelled) return;
          url = URL.createObjectURL(blob);
          video.src = url; video.controls = true; video.autoplay = false;
          setStateValue('clip', {mime:blob.type, base64:String(reader.result).split(',')[1]});
          status.textContent = 'วิดีโอพร้อมแล้ว กดวิเคราะห์ข้อมูลที่เลือกด้านล่าง';
          start.disabled = false;
        };
        reader.onerror = () => { status.textContent = 'อ่านวิดีโอไม่สำเร็จ กรุณาอัดใหม่'; start.disabled = false; };
        reader.readAsDataURL(blob);
      };
      recorder.start(250);
      setStateValue('clip', null);
      stop.disabled = false; clear.disabled = false;
      status.textContent = 'กำลังอัด 0 / 30 วินาที';
      timer = setInterval(() => {
        elapsed += 1;
        status.textContent = `กำลังอัด ${elapsed} / 30 วินาที`;
        if (elapsed >= 30) finish();
      }, 1000);
    } catch (error) {
      release(); start.disabled = false; clear.disabled = false;
      const messages = {NotAllowedError:'ไม่ได้รับอนุญาตให้ใช้กล้อง กรุณาอนุญาตในเบราว์เซอร์แล้วลองใหม่', NotFoundError:'ไม่พบกล้องที่เชื่อมต่อ', NotReadableError:'เปิดกล้องไม่ได้ กรุณาปิดแอปอื่นที่ใช้กล้องอยู่'};
      status.textContent = messages[error.name] || `เปิดกล้องไม่ได้: ${error.message}`;
    }
  };
  return () => { disposed = true; cancelled = true; finish(); release(); revoke(); };
}

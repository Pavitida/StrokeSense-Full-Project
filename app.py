"""StrokeSense guided local research demo. Browser media stays session-scoped."""
import os
import time
from pathlib import Path
import requests
import streamlit as st
from src.capture import recorded_video

st.set_page_config(page_title="StrokeSense · Research", page_icon="🧠", layout="wide")
st.markdown('''<style>
.block-container{max-width:1060px;padding-top:2rem;padding-bottom:4rem}
h1,h2,h3{letter-spacing:-.025em} .hero{padding:2rem;border:1px solid #3c6275;border-radius:22px;background:linear-gradient(120deg,#102b3b,#173d49);color:#fff;margin-bottom:1.3rem}
.hero p{max-width:660px;line-height:1.8;color:#d5e8ee}.eyebrow{color:#8cd4cc;font-size:.8rem;letter-spacing:.14em}
div[data-testid="stMetric"]{border:1px solid #53646b;border-radius:14px;padding:1rem}
@media(max-width:640px){.block-container{padding:1rem}.hero{padding:1.2rem}h1{font-size:2rem!important}}
</style>''', unsafe_allow_html=True)

PAGES = ["Home", "วิธีใช้และความยินยอม", "Face", "Speech", "Movement", "Result", "ช่วยเหลือ / สิทธิ์อุปกรณ์"]
EMERGENCY = "หากมีหน้าเบี้ยว แขนขาอ่อนแรง หรือพูดผิดปกติเกิดขึ้นฉับพลัน ให้โทร 1669 ทันที ไม่ต้องรอผล AI"
ROOT = Path(__file__).resolve().parent
TTL = 15 * 60
st.session_state.setdefault("media", {})
st.session_state.setdefault("consented", False)
st.session_state.setdefault("last_activity", time.time())
st.session_state.setdefault("nav", "Home")
if "pending_nav" in st.session_state:
    st.session_state.nav = st.session_state.pop("pending_nav")
st.session_state.setdefault("generation", 0)

def go(page):
    st.session_state.pending_nav = page
    st.session_state.last_activity = time.time()

def reset():
    generation = st.session_state.get("generation", 0) + 1
    for key in list(st.session_state):
        del st.session_state[key]
    st.session_state.update(media={}, consented=False, pending_nav="Home", last_activity=time.time(), generation=generation)

@st.fragment(run_every=60)
def expiry():
    if time.time() - st.session_state.last_activity > TTL and (st.session_state.media or st.session_state.get("result")):
        reset()
        st.rerun()
expiry()

with st.sidebar:
    st.title("🧠 StrokeSense")
    st.caption("LOCAL RESEARCH PROTOTYPE")
    page = st.radio("ขั้นตอน", PAGES, key="nav", on_change=lambda: st.session_state.update(last_activity=time.time()))
    st.divider()
    st.caption("ข้อมูลที่เตรียมไว้")
    for name, key in [("Face", "image"), ("Speech", "audio"), ("Movement", "video")]:
        st.write(("✓ " if key in st.session_state.media else "○ ") + name)
    st.button("ยกเลิกและลบข้อมูลเซสชัน", on_click=reset, use_container_width=True)
    st.caption("ไฟล์และผลถูกล้างจากเซสชันเมื่อยกเลิก หรือไม่มีการใช้งาน 15 นาที ขณะหน้าเว็บยังเชื่อมต่ออยู่")

st.warning(EMERGENCY)
st.caption("ต้นแบบวิจัย • ไม่ใช่การวินิจฉัย • คะแนนไม่ใช่เปอร์เซ็นต์โอกาสเป็นสโตรก")

def preview(kind, content, filename):
    if kind == "image":
        st.image(content, width=320)
    elif kind == "audio":
        st.audio(content, format="audio/wav")
    else:
        st.video(content, format="video/webm" if filename.endswith(".webm") else "video/mp4")

def save_input(kind, filename, content, next_page):
    limit = {"image": 5, "audio": 10, "video": 20}[kind] * 1024**2
    if not content or len(content) > limit:
        st.error(f"ไฟล์ว่างหรือใหญ่เกิน {limit // 1024**2} MB")
        return
    st.session_state.media[kind] = (filename, content)
    st.session_state.pop("result", None)
    go(next_page)
    st.rerun()

if page == "Home":
    st.markdown('''<div class="hero"><div class="eyebrow">STROKESENSE · RESEARCH DEMO</div>
    <h1>ทดลอง AI ผ่านใบหน้า<br>เสียงพูด และการเคลื่อนไหว</h1>
    <p>สำรวจการทำงานของโมเดล 3 ส่วนด้วยกล้อง ไมโครโฟน หรือไฟล์ของคุณ ดูผลแยกแต่ละส่วนพร้อมข้อจำกัด ไม่ใช้แทนการประเมินของแพทย์</p></div>''', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, title, description in zip(cols, ["01 · Face", "02 · Speech", "03 · Movement"], ["ภาพใบหน้าและความไม่สมมาตร", "ลักษณะเสียงพูดจากการอ่านประโยค", "จุดร่างกายจากวิดีโอการเคลื่อนไหว"]):
        with col, st.container(border=True):
            st.subheader(title)
            st.write(description)
    st.button("เริ่มทดลอง", type="primary", on_click=go, args=("วิธีใช้และความยินยอม",))
    st.info("โมเดลยังไม่ผ่านการตรวจสอบทางคลินิก และผลการเคลื่อนไหวมีความคลาดเคลื่อนสูง ไม่ควรใช้ผลเพื่อตัดสินว่าปลอดภัยหรือไม่ต้องพบแพทย์")
elif page == "วิธีใช้และความยินยอม":
    st.title("ก่อนเริ่มทดลอง")
    st.markdown("""1. ใช้แสงด้านหน้า กล้องระดับสายตา และพื้นที่เงียบ
2. ถ่ายภาพยิ้ม อัดเสียงอ่านประโยค และอัดวิดีโอให้เห็นแขนทั้งสองข้าง
3. ตรวจตัวอย่างก่อนเลือกใช้ แล้วส่งประมวลผลเมื่อพร้อม สามารถทดสอบเฉพาะบางส่วนได้

**ข้อจำกัด:** Face ประเมินความผิดปกติของใบหน้า, Speech จำแนกลักษณะเสียงพูด, Movement เรียนรู้ความครบถ้วนของท่าฟื้นฟู ไม่ใช่การตรวจแขนอ่อนแรงจากสโตรก ภาษาและท่าที่ต่างจากข้อมูลฝึกอาจให้ผลคลาดเคลื่อน ไม่ได้ตรวจว่าอ่านถูกประโยคหรือทำท่าถูกต้อง""")
    with st.container(border=True):
        st.subheader("ข้อมูลของคุณ")
        st.write("เมื่อกดบันทึก เบราว์เซอร์จะขอสิทธิ์กล้อง/ไมค์ ข้อมูลที่บันทึกจะส่งเข้าเซสชันเว็บบนเครื่องนี้ และส่งไป API ใน Ubuntu เมื่อกดวิเคราะห์ ไม่มีการส่งไปบริการ AI ภายนอก")
        st.write("ไฟล์ชั่วคราวของ API ลบหลังจบคำขอทั้งกรณีสำเร็จและผิดพลาด ผลอยู่เฉพาะเซสชัน ไม่มีฐานข้อมูลผู้ใช้ ไม่มีการบันทึกสื่อเพื่อวิจัย ปุ่มยกเลิกล้างข้อมูลของเซสชันนี้")
        consent = st.checkbox("ฉันเข้าใจข้อจำกัดและยินยอมให้ประมวลผลข้อมูลเพื่อทดลองครั้งนี้", value=st.session_state.consented)
        st.checkbox("ยินยอมใช้ข้อมูลเพื่อวิจัย (ยังไม่เปิดให้ใช้งาน)", value=False, disabled=True)
        st.caption("ความยินยอมทดสอบไม่ใช่ความยินยอมวิจัย ระบบไม่เรียก Data Collection และไม่มีการเก็บข้อมูลวิจัยอัตโนมัติ")
    if st.button("ยืนยันและไป Face", type="primary", disabled=not consent):
        st.session_state.consented = True
        go("Face")
        st.rerun()
elif page in {"Face", "Speech", "Movement"}:
    st.title({"Face":"01 · ภาพใบหน้า", "Speech":"02 · เสียงพูด", "Movement":"03 · การเคลื่อนไหว"}[page])
    if not st.session_state.consented:
        st.info("กรุณาอ่านคำแนะนำและให้ความยินยอมก่อนเปิดอุปกรณ์")
        st.button("ไปหน้าความยินยอม", on_click=go, args=("วิธีใช้และความยินยอม",))
        st.stop()
    kind = {"Face":"image", "Speech":"audio", "Movement":"video"}[page]
    next_page = {"Face":"Speech", "Speech":"Movement", "Movement":"Result"}[page]
    st.progress({"Face":1/3, "Speech":2/3, "Movement":1.0}[page], text="เตรียมข้อมูล " + page)
    guide = {
        "Face":"มองตรงที่กล้อง ให้เห็นใบหน้าทั้งหมดและยิ้มตามธรรมชาติ หลีกเลี่ยงย้อนแสง รองรับ JPG/PNG ≤5 MB, 128–4096 พิกเซลต่อด้าน",
        "Speech":"อ่านด้วยเสียงปกติ: ‘วันนี้ฉันกำลังทดลองอ่านประโยคเพื่อบันทึกเสียงพูด’ รองรับ WAV 1–30 วินาที ≤10 MB ประโยคนี้เป็นแนวทางบันทึก ยังไม่มีการตรวจความถูกต้องของคำหรือการรับรองกับภาษาไทย",
        "Movement":"วางกล้องให้เห็นใบหน้า ลำตัว และแขนทั้งสองข้าง ยกแขนตามความสามารถแล้วลดลง ไม่ฝืนหากเจ็บ อัดประมาณ 5–10 วินาที รองรับ MP4/WebM/MOV 1–30 วินาที ≤20 MB, ไม่เกิน 1080p/60 FPS ท่านี้เป็นการสาธิต ไม่ใช่การทดสอบทางคลินิกที่ผ่านการยืนยัน",
    }
    st.write(guide[page])
    source = st.radio("นำเข้าข้อมูล", ["อัปโหลดไฟล์", "กล้อง / ไมโครโฟน"], horizontal=True, key=f"source_{kind}")
    data = None
    filename = ""
    generation = st.session_state.generation
    if source == "อัปโหลดไฟล์":
        upload = st.file_uploader("เลือกไฟล์", type={"image":["jpg","jpeg","png"], "audio":["wav"], "video":["mp4","webm","mov"]}[kind], key=f"upload_{kind}_{generation}")
        if upload is not None:
            data, filename = upload.getvalue(), upload.name
    elif kind == "image":
        upload = st.camera_input("ถ่ายภาพใบหน้า", key=f"camera_{generation}")
        if upload is not None:
            data, filename = upload.getvalue(), "face.jpg"
    elif kind == "audio":
        upload = st.audio_input("กดไมค์เพื่อเริ่มอ่าน แล้วกดหยุดเมื่อจบ", sample_rate=16000, key=f"microphone_{generation}")
        if upload is not None:
            data, filename = upload.getvalue(), "speech.wav"
    else:
        try:
            recorded = recorded_video()
            if recorded:
                data, extension = recorded
                filename = "movement" + extension
        except ValueError as error:
            st.error(str(error))
    if data:
        st.subheader("ตรวจตัวอย่างก่อนเลือกใช้")
        preview(kind, data, filename)
        if st.button("ใช้ข้อมูลนี้และไปต่อ", type="primary"):
            save_input(kind, filename, data, next_page)
    elif kind in st.session_state.media:
        st.success("มีข้อมูลที่เลือกไว้แล้ว")
        filename, content = st.session_state.media[kind]
        preview(kind, content, filename)
    cols = st.columns(2)
    with cols[0]:
        if st.button("ลบข้อมูลส่วนนี้ / บันทึกใหม่"):
            st.session_state.media.pop(kind, None)
            st.session_state.pop("result", None)
            st.session_state.generation += 1
            st.session_state.pop("movement_recorder", None)
            st.rerun()
    with cols[1]:
        st.button("ไปขั้นตอนถัดไป", on_click=go, args=(next_page,))
    st.button("กล้องหรือไมค์ใช้ไม่ได้", on_click=go, args=("ช่วยเหลือ / สิทธิ์อุปกรณ์",))
elif page == "Result":
    st.title("สรุปผลการทดลอง")
    if not st.session_state.consented:
        st.info("ต้องยินยอมประมวลผลก่อนเริ่มทดลอง")
    else:
        for kind, (filename, content) in st.session_state.media.items():
            with st.expander({"image":"รูปที่เลือก", "audio":"เสียงที่เลือก", "video":"วิดีโอที่เลือก"}[kind]):
                preview(kind, content, filename)
        st.caption("วิเคราะห์ได้ทีละส่วนหรือทั้ง 3 ส่วน ข้อมูลที่ยังไม่ได้เลือกจะแสดง incomplete")
        if st.button("ส่งข้อมูลและวิเคราะห์", type="primary", disabled=not st.session_state.media):
            token = os.environ.get("STROKESENSE_API_TOKEN")
            if not token:
                st.session_state["error"] = "กรุณาเปิดเว็บด้วย bash run-ubuntu.sh เพื่อเริ่ม API และเว็บพร้อมกัน"
                go("ช่วยเหลือ / สิทธิ์อุปกรณ์")
                st.rerun()
            try:
                with st.spinner("กำลังประมวลผล โปรดรอและอย่าส่งซ้ำ..."):
                    files = {key:(filename, content, "application/octet-stream") for key,(filename,content) in st.session_state.media.items()}
                    response = requests.post(os.environ.get("STROKESENSE_API_URL", "http://127.0.0.1:8502") + "/api/analyze", headers={"Authorization":"Bearer " + token}, data={"consent":"true"}, files=files, timeout=(5,600))
                    payload = response.json()
                    if not response.ok:
                        raise ValueError(payload.get("reason", "ส่งข้อมูลไม่สำเร็จ"))
                    st.session_state.result = payload
                    st.session_state.media = {}
                    st.session_state.last_activity = time.time()
                    st.rerun()
            except requests.RequestException:
                st.session_state.error = "เชื่อมต่อ Backend ไม่สำเร็จหรือหมดเวลารอ ตรวจว่า Ubuntu และ API ยังทำงานอยู่ แล้วลองใหม่"
                go("ช่วยเหลือ / สิทธิ์อุปกรณ์")
                st.rerun()
            except ValueError as error:
                st.session_state.error = str(error)
                go("ช่วยเหลือ / สิทธิ์อุปกรณ์")
                st.rerun()
        result = st.session_state.get("result")
        if result:
            st.write("สถานะรวม:", result["status"])
            for name, description in [("face","คะแนนกลุ่มความผิดปกติของใบหน้า"),("speech","คะแนนกลุ่มลักษณะเสียง dysarthria"),("movement","คะแนนกลุ่มการเคลื่อนไหวไม่ครบตามโมเดลฟื้นฟู")]:
                item = result.get(name, {})
                with st.container(border=True):
                    st.subheader(name.title())
                    st.caption("สถานะ: " + item.get("status", "incomplete"))
                    if item.get("status") == "success":
                        st.metric(description, f"{item['score']:.3f}")
                        st.caption("ค่าจำแนก 0–1 ไม่ใช่ความแม่นยำหรือโอกาสเป็นโรค ค่าต่ำไม่ยืนยันว่าปกติ")
                    else:
                        st.info(item.get("reason", "ไม่พบข้อมูลที่เหมาะสม กรุณาบันทึกใหม่"))
            fusion = result.get("fusion")
            if fusion and fusion.get("status") == "success":
                st.metric("คะแนนรวมเชิงทดลอง — ไม่ใช่โอกาสเป็นสโตรก", f"{fusion['score']:.3f}")
            else:
                st.info("Fusion: " + (fusion.get("status", "incomplete") if fusion else "incomplete") + " — ยังไม่แสดงคะแนนรวม เพราะข้อมูลหรือโมเดลยังไม่พร้อม")
            st.warning("การทดลองครั้งก่อนพบ Movement ผิดหลายตัวอย่าง และ CPU/GPU ให้ผลต่างกัน จึงยังไม่พร้อมใช้ประเมินสุขภาพ")
    st.button("เริ่มทดสอบใหม่และลบผล", on_click=reset)
    st.button("กลับ Home", on_click=go, args=("Home",))
else:
    st.title("ช่วยเหลือและสิทธิ์อุปกรณ์")
    if st.session_state.get("error"):
        st.error(st.session_state.error)
    st.markdown("""**ปฏิเสธสิทธิ์กล้อง/ไมค์:** เปิดการตั้งค่าเว็บไซต์ที่ไอคอนข้าง URL แล้วอนุญาต Camera/Microphone จากนั้นโหลดหน้าใหม่ หรือเลือกอัปโหลดไฟล์แทน

**ไม่พบอุปกรณ์ / กล้องไม่ว่าง:** ตรวจการเชื่อมต่อและปิดโปรแกรมอื่นที่ใช้กล้อง ลอง Chrome หรือ Edge หากเบราว์เซอร์ในแอปไม่รองรับ

**ไฟล์ไม่ผ่าน:** ใช้ JPG/PNG ≤5 MB, WAV ≤10 MB หรือ MP4/WebM/MOV ≤20 MB เสียงและวิดีโอต้องยาว 1–30 วินาที เปลี่ยนนามสกุลอย่างเดียวไม่ได้ ต้องเป็นไฟล์ที่ถอดรหัสได้จริง

**อัปโหลดหรือประมวลผลล้มเหลว:** ตรวจว่า Ubuntu ยังทำงานอยู่ เปิดด้วย `bash run-ubuntu.sh` แล้วลองใหม่หลังคำขอก่อนหน้าจบ

**การลบข้อมูล:** ใช้ปุ่มยกเลิกในแถบด้านข้างเพื่อล้างสื่อ ผล และความยินยอม กล้องวิดีโอจะปิดหลังหยุดบันทึกหรือออกจากหน้านั้น หากเกิดโปรเซสล่ม ผู้ดูแลต้องล้างโฟลเดอร์ชั่วคราวตามคู่มือ

**การใช้งานผ่านเครือข่าย:** รุ่นนี้เปิดเฉพาะ localhost การเปิดให้ผู้อื่นเข้าถึงต้องเพิ่ม HTTPS และระบบยืนยันตัวตนก่อน ไม่เปิดพอร์ต API สู่ภายนอก""")
    st.button("กลับไปตรวจข้อมูล", on_click=go, args=("Result",))
    st.button("เริ่มใหม่และล้างข้อมูล", on_click=reset)

import streamlit as st
import json
import os
import time
from contextlib import contextmanager
from json import JSONDecodeError
from pathlib import Path

_USER_DATA_DIR = Path.home() / ".idopnetwork"
_USER_DATA_FILE = _USER_DATA_DIR / "users.json"
_USER_DATA_LOCK_DIR = _USER_DATA_DIR / ".users.lock"
_LOCK_TIMEOUT_SECONDS = 5


def _default_users():
    return {
        "admin": {
            "password": "admin",
            "real_name": "Administrator",
            "phone": "",
            "organization": "",
            "research_direction": "",
        }
    }


@contextmanager
def _user_file_lock():
    """Best-effort cross-platform lock for the local JSON user store."""
    _USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    while True:
        try:
            _USER_DATA_LOCK_DIR.mkdir()
            break
        except FileExistsError:
            if time.monotonic() - start >= _LOCK_TIMEOUT_SECONDS:
                raise TimeoutError("用户数据文件正忙，请稍后重试")
            time.sleep(0.05)

    try:
        yield
    finally:
        try:
            _USER_DATA_LOCK_DIR.rmdir()
        except FileNotFoundError:
            pass


def _read_users_file():
    if not _USER_DATA_FILE.exists():
        return {}
    try:
        with open(_USER_DATA_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)
    except JSONDecodeError:
        backup_file = _USER_DATA_FILE.with_suffix(f".corrupt-{int(time.time())}.json")
        os.replace(_USER_DATA_FILE, backup_file)
        return {}

    return users if isinstance(users, dict) else {}


def _write_users_file(users):
    tmp_file = _USER_DATA_FILE.with_suffix(".json.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)
    os.replace(tmp_file, _USER_DATA_FILE)


def _normalize_username(username):
    return username.strip()


def _ensure_data_dir():
    """确保数据目录存在，并创建默认管理员账号。"""
    _USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _USER_DATA_FILE.exists():
        _write_users_file(_default_users())


def load_users():
    """读取用户信息"""
    with _user_file_lock():
        _ensure_data_dir()
        return _read_users_file()


def save_users(users):
    """保存用户信息"""
    with _user_file_lock():
        _write_users_file(users)


def create_user(username, user_info):
    """Create a user from the latest on-disk data to avoid overwriting peers."""
    username = _normalize_username(username)
    with _user_file_lock():
        _ensure_data_dir()
        users = _read_users_file()
        if username in users:
            return False
        users[username] = user_info
        _write_users_file(users)
        return True


def show_login_ui():
    """右上角登录按钮"""
    col_title, col_btn = st.columns([8.5, 1.5])
    with col_btn:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🔑 登录 / 注册", type="primary", use_container_width=True):
            show_auth_modal()


@st.dialog("🔐 统一身份认证中心", width="large")
def show_auth_modal():
    """弹窗内部：登录与详细信息注册"""
    tab_login, tab_register = st.tabs(["🔑 登录已有账户", "📝 注册新账户"])

    with tab_login:
        log_user = st.text_input("用户名", key="log_user")
        log_pwd = st.text_input("密码", type="password", key="log_pwd")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("验证并进入系统", type="primary", use_container_width=True):
            log_user = _normalize_username(log_user)
            users = load_users()
            user_info = users.get(log_user)
            if user_info and isinstance(user_info, dict) and user_info.get("password") == log_pwd:
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = log_user
                st.session_state["user_details"] = user_info
                st.rerun()
            elif user_info == log_pwd:
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = log_user
                st.rerun()
            else:
                st.error("用户名或密码错误")

    with tab_register:
        st.markdown("##### 请填写您的真实研究信息")
        reg_user = st.text_input("用户名 *", placeholder="用于登录的唯一ID", key="reg_user")
        reg_pwd = st.text_input("设置密码 *", type="password", key="reg_pwd")
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            reg_real_name = st.text_input("真实姓名", placeholder="张三", key="reg_real_name")
            reg_phone = st.text_input("电话号", placeholder="138XXXXXXXX", key="reg_phone")
        with col_reg2:
            reg_org = st.text_input("所属单位", placeholder="XX大学/XX研究院", key="reg_org")
            reg_field = st.text_input("研究方向", placeholder="复杂网络/生物信息等", key="reg_field")
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("ℹ️ **声明**：您填写的个人信息仅用于学术交流及课题组内部成员身份核验，平台将严格保护您的隐私。")

        if st.button("提交注册申请", use_container_width=True, type="primary"):
            reg_user = _normalize_username(reg_user)
            if not reg_user or not reg_pwd:
                st.warning("请至少填写用户名和密码")
            else:
                created = create_user(reg_user, {
                    "password": reg_pwd,
                    "real_name": reg_real_name,
                    "phone": reg_phone,
                    "organization": reg_org,
                    "research_direction": reg_field,
                })
                if created:
                    st.success("🎉 注册成功！请切换到『登录』页进行验证。")
                else:
                    st.error("该用户名已存在，请更换")

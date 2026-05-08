import streamlit as st
import json
import os

# 用户数据保存路径
USER_DATA_FILE = "backend/users.json"

def load_users():
    """读取用户信息"""
    if not os.path.exists(USER_DATA_FILE):
        return {}
    with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    """保存用户信息"""
    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

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
    users = load_users()
    
    with tab_login:
        log_user = st.text_input("用户名", key="log_user")
        log_pwd = st.text_input("密码", type="password", key="log_pwd")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("验证并进入系统", type="primary", use_container_width=True):
            user_info = users.get(log_user)
            # 兼容旧版本和新版本的字典结构
            if user_info and isinstance(user_info, dict) and user_info.get("password") == log_pwd:
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = log_user
                st.session_state["user_details"] = user_info
                st.rerun()
            elif user_info == log_pwd: # 兼容旧的字符串存法
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = log_user
                st.rerun()
            else:
                st.error("用户名或密码错误")
                
    with tab_register:
        st.markdown("##### 请填写您的真实研究信息")
        
        # 必填项
        reg_user = st.text_input("用户名 *", placeholder="用于登录的唯一ID", key="reg_user")
        reg_pwd = st.text_input("设置密码 *", type="password", key="reg_pwd")
        
        # 详细信息收集
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            reg_real_name = st.text_input("真实姓名", placeholder="张三", key="reg_real_name")
            reg_phone = st.text_input("电话号", placeholder="138XXXXXXXX", key="reg_phone")
        with col_reg2:
            reg_org = st.text_input("所属单位", placeholder="XX大学/XX研究院", key="reg_org")
            reg_field = st.text_input("研究方向", placeholder="复杂网络/生物信息等", key="reg_field")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 学术免责声明
        st.info("ℹ️ **声明**：您填写的个人信息仅用于学术交流及课题组内部成员身份核验，平台将严格保护您的隐私。")
        
        if st.button("提交注册申请", use_container_width=True, type="primary"):
            if not reg_user or not reg_pwd:
                st.warning("请至少填写用户名和密码")
            elif reg_user in users:
                st.error("该用户名已存在，请更换")
            else:
                # 🌟 以字典形式存储所有详细信息
                users[reg_user] = {
                    "password": reg_pwd,
                    "real_name": reg_real_name,
                    "phone": reg_phone,
                    "organization": reg_org,
                    "research_direction": reg_field
                }
                save_users(users)
                st.success("🎉 注册成功！请切换到『登录』页进行验证。")
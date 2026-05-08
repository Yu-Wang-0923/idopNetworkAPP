import streamlit as st
import json
import os

# 我们用来存账号密码的“小后台数据库”文件
USER_DATA_FILE = "backend/users.json"

def load_users():
    """读取本地的账号数据，如果没有就建个空的"""
    if not os.path.exists(USER_DATA_FILE):
        return {}
    with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    """把新注册的账号保存到本地文件里"""
    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def show_login_ui():
    """显示登录和注册界面"""
    st.markdown("<br><br>", unsafe_allow_html=True) # 往下挪一点，好看些
    
    # 用两个标签页把“登录”和“注册”分开
    tab_login, tab_register = st.tabs(["🔑 登录", "📝 创建新账户"])
    
    users = load_users()
    
    # --- 登录模块 ---
    with tab_login:
        log_user = st.text_input("用户名", key="log_user")
        log_pwd = st.text_input("密码", type="password", key="log_pwd")
        
        if st.button("登录", type="primary"):
            if log_user in users and users[log_user] == log_pwd:
                # 登录成功，把状态存进 Streamlit 的“记忆(session_state)”里
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = log_user
                st.success("登录成功！页面即将跳转...")
                st.rerun() # 强制刷新页面，让侧边栏的功能冒出来
            else:
                st.error("用户名或密码错误，请重试！")
                
    # --- 注册模块 ---
    with tab_register:
        reg_user = st.text_input("设置用户名", key="reg_user")
        reg_pwd = st.text_input("设置密码", type="password", key="reg_pwd")
        reg_pwd2 = st.text_input("确认密码", type="password", key="reg_pwd2")
        
        if st.button("立即注册"):
            if not reg_user or not reg_pwd:
                st.warning("用户名和密码不能为空！")
            elif reg_user in users:
                st.error("哎呀，这个用户名已经被注册啦，换一个吧！")
            elif reg_pwd != reg_pwd2:
                st.error("两次输入的密码不一致哦！")
            else:
                # 没问题的话，就把新用户存起来！
                users[reg_user] = reg_pwd
                save_users(users)
                st.success(f"🎉 注册成功！欢迎你，{reg_user}！现在你可以去左侧『登录』了！")
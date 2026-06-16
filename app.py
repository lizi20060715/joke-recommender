import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

def load_data():
    """加载评分数据和笑话文本数据"""
    ratings_df = pd.read_csv('processed_ratings.csv')
    
    jokes_raw = pd.read_excel('Dataset4JokeSet.xlsx', header=None)
    jokes_list = []
    
    if len(jokes_raw.columns) > 0:
        first_joke = jokes_raw.columns[0]
        jokes_list.append(first_joke)
        
        for _, row in jokes_raw.iterrows():
            joke_text = row[0]
            if pd.notna(joke_text):
                jokes_list.append(joke_text)
    
    jokes_df = pd.DataFrame({
        'joke_id': range(1, len(jokes_list) + 1),
        'joke_text': jokes_list
    })
    
    return ratings_df, jokes_df

def filter_valid_jokes(ratings_df, jokes_df):
    """过滤掉无任何评分的无效笑话"""
    rated_joke_ids = ratings_df['joke_id'].unique()
    filtered_ratings = ratings_df[ratings_df['joke_id'].isin(rated_joke_ids)]
    filtered_jokes = jokes_df[jokes_df['joke_id'].isin(rated_joke_ids)]
    
    st.sidebar.write(f"**数据统计**")
    st.sidebar.write(f"- 原始笑话数: {len(jokes_df)}")
    st.sidebar.write(f"- 有评分的笑话数: {len(filtered_jokes)}")
    st.sidebar.write(f"- 评分记录数: {len(filtered_ratings)}")
    st.sidebar.write(f"- 用户数: {filtered_ratings['user_id'].nunique()}")
    
    return filtered_ratings, filtered_jokes

def build_item_similarity_matrix(ratings_df):
    """使用Item-Item协同过滤和余弦相似度构建相似度矩阵"""
    user_joke_matrix = ratings_df.pivot_table(
        index='user_id',
        columns='joke_id',
        values='rating',
        fill_value=0
    )
    
    joke_ids = user_joke_matrix.columns.tolist()
    similarity_matrix = cosine_similarity(user_joke_matrix.T)
    
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=joke_ids,
        columns=joke_ids
    )
    
    return similarity_df

def recommend_jokes(similarity_df, rated_jokes, ratings, top_n=5):
    """根据用户评分加权计算，推荐Top N笑话"""
    valid_indices = [i for i, joke_id in enumerate(rated_jokes) if joke_id in similarity_df.index]
    valid_jokes = [rated_jokes[i] for i in valid_indices]
    valid_ratings = [ratings[i] for i in valid_indices]
    
    if not valid_jokes:
        return [], []
    
    recommendation_scores = pd.Series(0, index=similarity_df.index)
    
    for joke_id, rating in zip(valid_jokes, valid_ratings):
        similarities = similarity_df.loc[joke_id]
        recommendation_scores += similarities * rating
    
    recommendation_scores = recommendation_scores.drop(valid_jokes, errors='ignore')
    recommendation_scores = recommendation_scores.sort_values(ascending=False)
    
    recommended_jokes = recommendation_scores.head(top_n).index.tolist()
    scores = recommendation_scores.head(top_n).values.tolist()
    
    return recommended_jokes, scores

def calculate_satisfaction(recommended_ratings):
    """计算用户对推荐结果的满意度"""
    if not recommended_ratings:
        return 0, 0
    
    avg_rating = np.mean(recommended_ratings)
    min_rating = -10
    max_rating = 10
    normalized_score = (avg_rating - min_rating) / (max_rating - min_rating)
    satisfaction = min(max(normalized_score * 100, 0), 100)
    
    return satisfaction, avg_rating

def main():
    """Streamlit主应用函数"""
    st.set_page_config(
        page_title="笑话推荐系统",
        page_icon="😄",
        layout="wide"
    )
    
    st.title("😄 智能笑话推荐系统")
    st.markdown("---")
    
    if 'stage' not in st.session_state:
        st.session_state.stage = 'rating'
    if 'selected_jokes' not in st.session_state:
        st.session_state.selected_jokes = []
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    if 'recommended_jokes' not in st.session_state:
        st.session_state.recommended_jokes = []
    if 'recommendation_scores' not in st.session_state:
        st.session_state.recommendation_scores = []
    if 'feedback_ratings' not in st.session_state:
        st.session_state.feedback_ratings = {}
    if 'satisfaction' not in st.session_state:
        st.session_state.satisfaction = 0
    
    with st.spinner("正在加载数据..."):
        ratings_df, jokes_df = load_data()
        ratings_df, jokes_df = filter_valid_jokes(ratings_df, jokes_df)
        similarity_df = build_item_similarity_matrix(ratings_df)
    
    if st.session_state.stage == 'rating':
        st.subheader("📝 请为以下笑话评分")
        st.markdown("请根据您的喜好为3个随机笑话评分，我们将根据您的偏好推荐更多笑话！")
        
        valid_joke_ids = similarity_df.index.tolist()
        
        if not st.session_state.selected_jokes:
            st.session_state.selected_jokes = random.sample(valid_joke_ids, min(3, len(valid_joke_ids)))
        
        for idx, joke_id in enumerate(st.session_state.selected_jokes, 1):
            joke_text = jokes_df[jokes_df['joke_id'] == joke_id]['joke_text'].iloc[0]
            
            st.markdown(f"### 笑话 {idx}")
            st.write(f'"{joke_text}"')
            
            rating = st.slider(
                f"请为笑话 {idx} 评分",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                key=f"rating_{joke_id}"
            )
            st.session_state.user_ratings[joke_id] = rating
            st.markdown("---")
        
        if st.button("🎯 开始推荐", key="submit_ratings"):
            st.session_state.stage = 'recommend'
            st.experimental_rerun()
    
    elif st.session_state.stage == 'recommend':
        rated_jokes = list(st.session_state.user_ratings.keys())
        ratings = list(st.session_state.user_ratings.values())
        
        recommended_jokes, scores = recommend_jokes(
            similarity_df, 
            rated_jokes, 
            ratings, 
            top_n=5
        )
        
        st.session_state.recommended_jokes = recommended_jokes
        st.session_state.recommendation_scores = scores
        
        st.subheader("🎉 为您推荐的Top 5笑话")
        st.markdown("根据您的评分偏好，我们为您推荐以下笑话：")
        
        for idx, (joke_id, score) in enumerate(zip(recommended_jokes, scores), 1):
            joke_text = jokes_df[jokes_df['joke_id'] == joke_id]['joke_text'].iloc[0]
            
            st.markdown(f"### 推荐 {idx} (相似度分数: {score:.2f})")
            st.write(f'"{joke_text}"')
            
            feedback_rating = st.slider(
                f"您喜欢这个推荐吗？",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                key=f"feedback_{joke_id}"
            )
            st.session_state.feedback_ratings[joke_id] = feedback_rating
            st.markdown("---")
        
        if st.button("📊 计算满意度", key="calculate_satisfaction"):
            feedback_values = list(st.session_state.feedback_ratings.values())
            satisfaction, avg_rating = calculate_satisfaction(feedback_values)
            st.session_state.satisfaction = satisfaction
            st.session_state.avg_rating = avg_rating
            st.session_state.stage = 'feedback'
            st.experimental_rerun()
        
        if st.button("🔄 返回重新评分", key="back_to_rating"):
            st.session_state.stage = 'rating'
            st.session_state.selected_jokes = []
            st.session_state.user_ratings = {}
            st.session_state.recommended_jokes = []
            st.session_state.recommendation_scores = []
            st.session_state.feedback_ratings = {}
            st.experimental_rerun()
    
    elif st.session_state.stage == 'feedback':
        st.subheader("📈 推荐满意度报告")
        
        st.markdown("### 您的初始评分")
        for joke_id, rating in st.session_state.user_ratings.items():
            joke_text = jokes_df[jokes_df['joke_id'] == joke_id]['joke_text'].iloc[0]
            st.write(f"- **评分**: {rating} | {joke_text[:50]}...")
        
        st.markdown("---")
        
        st.markdown("### 推荐结果评分")
        for joke_id, rating in st.session_state.feedback_ratings.items():
            joke_text = jokes_df[jokes_df['joke_id'] == joke_id]['joke_text'].iloc[0]
            st.write(f"- **评分**: {rating} | {joke_text[:50]}...")
        
        st.markdown("---")
        
        st.markdown("### 满意度统计")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="平均评分",
                value=f"{st.session_state.avg_rating:.2f}",
                delta="满分: 10"
            )
        with col2:
            st.metric(
                label="满意度",
                value=f"{st.session_state.satisfaction:.1f}%",
                delta="目标: 100%"
            )
        
        st.progress(st.session_state.satisfaction / 100)
        
        if st.session_state.satisfaction >= 80:
            st.success("🎉 非常满意！感谢您的使用！")
        elif st.session_state.satisfaction >= 60:
            st.info("😊 满意！我们会继续努力！")
        elif st.session_state.satisfaction >= 40:
            st.warning("😐 一般，我们会改进推荐算法！")
        else:
            st.error("😢 不满意，我们会认真改进！")
        
        st.markdown("---")
        
        if st.button("🔄 再来一次", key="restart"):
            st.session_state.stage = 'rating'
            st.session_state.selected_jokes = []
            st.session_state.user_ratings = {}
            st.session_state.recommended_jokes = []
            st.session_state.recommendation_scores = []
            st.session_state.feedback_ratings = {}
            st.session_state.satisfaction = 0
            st.experimental_rerun()

if __name__ == "__main__":
    main()

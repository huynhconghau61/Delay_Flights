import streamlit as st
import base64
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import shap
def set_bg(bg_image_path):
    # Đọc file hình ảnh và chuyển đổi nó thành chuỗi base64
    with open(bg_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # Thiết lập CSS để dãn hình nền và thêm hoạt hình slide ngang từ phải qua trái
    bg_image_style = f"""
    <style>
    @keyframes slideBackground {{
        from {{ background-position: 0 0; }}  # Bắt đầu từ bên trái
        to {{ background-position: 100% 0; }}  # Kết thúc bên phải
    }}
    .stApp {{
        background: url("data:image/jpeg;base64,{encoded_string}");
        background-size: 120% 110%;  # Đặt kích thước lớn gấp đôi chiều ngang để hiệu ứng trông mượt mà 170 140
        background-repeat: no-repeat;
        animation: slideBackground 20s linear infinite;
    }}
    </style>
    """
    st.markdown(bg_image_style, unsafe_allow_html=True)

# Hàm tải và xử lý dữ liệu
# Sử dụng @st.cache để cache kết quả của hàm loadData
#@st.cache
def loadData():
    flights = pd.read_csv('../data/flights.csv')
    airport = pd.read_csv('../data/airports.csv')
    variables_to_remove=["YEAR","FLIGHT_NUMBER","TAIL_NUMBER","DEPARTURE_TIME","TAXI_OUT","WHEELS_OFF","ELAPSED_TIME","AIR_TIME","WHEELS_ON","TAXI_IN","ARRIVAL_TIME","DIVERTED","CANCELLED","CANCELLATION_REASON","AIR_SYSTEM_DELAY", "SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","SCHEDULED_TIME","SCHEDULED_ARRIVAL"]
    flights.drop(variables_to_remove,axis=1,inplace= True)

    flights.loc[~flights.ORIGIN_AIRPORT.isin(airport.IATA_CODE.values),'ORIGIN_AIRPORT']='OTHER'
    flights.loc[~flights.DESTINATION_AIRPORT.isin(airport.IATA_CODE.values),'DESTINATION_AIRPORT']='OTHER'

    flights=flights.dropna()

    df=pd.DataFrame(flights)
    df['DAY_OF_WEEK']= df['DAY_OF_WEEK'].apply(str)
    df["DAY_OF_WEEK"].replace({"1":"SUNDAY", "2": "MONDAY", "3": "TUESDAY", "4":"WEDNESDAY", "5":"THURSDAY", "6":"FRIDAY", "7":"SATURDAY"},inplace=True)

    dums = ['AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT','DAY_OF_WEEK']
    df_cat=pd.get_dummies(df[dums],drop_first=True)

    var_to_remove=["DAY_OF_WEEK","AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT"]
    df.drop(var_to_remove,axis=1,inplace=True)

    data=pd.concat([df,df_cat],axis=1)
    final_data = data.sample(n=60000)
    return final_data
# Hàm tiền xử lý và chia dữ liệu
def preprocessing(final_data):
    X=final_data.drop("DEPARTURE_DELAY",axis=1)
    Y=final_data.DEPARTURE_DELAY
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return X_train,y_train,X
#Hàm xây dựng mô hình Random Forest Regressor
def rfg(X_train,y_train):
    reg_rf = RandomForestRegressor()
    reg_rf.fit(X_train,y_train)
    return reg_rf
def accept_data():
    # Áp dụng CSS để điều chỉnh kiểu chữ
    st.markdown("<style>.font {font-weight: bold;font-size: 20px; margin-top: 10px;  margin-bottom: -28px; color: black;}</style>", unsafe_allow_html=True)
    
    # Tạo 2 cột
    col1, col2 = st.columns(2)

    # Cột 1 - 5 hàng
    with col1:
        st.markdown('<p class="font">Enter month</p>', unsafe_allow_html=True)
        month = st.number_input("", min_value=1, max_value=12, key="month")
        
        st.markdown('<p class="font">Enter day</p>', unsafe_allow_html=True)
        day = st.number_input("", min_value=1, max_value=31, key="day")
        
        st.markdown('<p class="font">Enter scheduled departure</p>', unsafe_allow_html=True)
        sch_dept = st.number_input("", key="sch_dept")
        
        st.markdown('<p class="font">Enter distance in miles</p>', unsafe_allow_html=True)
        distance = st.number_input("", key="distance")
        
        st.markdown('<p class="font">Enter arrival delay</p>', unsafe_allow_html=True)
        arrival_delay = st.number_input("Enter negative value if early, positive if delayed", key="arrival_delay")

         # Thêm trường nhập số ghế bị chiếm
        st.markdown('<p class="font">Enter number of seats</p>', unsafe_allow_html=True)
        seats_occupied = st.number_input("", min_value=0, key="seats_occupied")

    # Cột 2 - 5 hàng
    with col2:
        st.markdown('<p class="font">Enter airline code</p>', unsafe_allow_html=True)
        airline_codes = ["XX", "B6", "AA", "WN", "EV", "DL", "UA", "OO", "NK", "HA", "US", "AS", "MQ", "F9", "VX"]
        formatted_airline_codes = ["AIRLINE_" + code for code in airline_codes]
        airline = st.selectbox("", options=formatted_airline_codes, index=0, key="airline")
        
        st.markdown('<p class="font">Enter origin airport code</p>', unsafe_allow_html=True)
        origin = st.text_input("", "ORIGIN_AIRPORT_XXX", key="origin")
        
        st.markdown('<p class="font">Enter destination airport code</p>', unsafe_allow_html=True)
        destination = st.text_input("", "DESTINATION_AIRPORT_XXX", key="destination")
        
        st.markdown('<p class="font">Enter day of week</p>', unsafe_allow_html=True)
        day_of_week_options = ["None", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY"]
        day_of_week = st.selectbox("", day_of_week_options, key="day_of_week")
        
        st.markdown('<p class="font">Lights status</p>', unsafe_allow_html=True)
        status_options = ["None", "Weather", "Airline", "Nas", "Security"]
        status_options1 = st.selectbox("", status_options, key="status_options1")
        # Thêm một trường nhập liệu hoặc widget tùy ý hoặc để trống

    return month, day, sch_dept, distance, arrival_delay, airline, origin, destination, day_of_week, seats_occupied, status_options1

# Hàm dự đoán
def prediction(X,month, day,sch_dept,distance,arrival_delay,airline,origin,destination,day_of_week,seats_occupied,status_options1,reg_rf):
    AIRLINE_index = np.where(X.columns==airline)
    ORIGIN_index = np.where(X.columns==origin)
    DESTINATION_index = np.where(X.columns==destination)
    DAY_OF_WEEK_index = np.where(X.columns==day_of_week)
    x= np.zeros(len(X.columns))
    x[0] = month
    x[1] = day
    x[2] = sch_dept
    x[3] = distance
    x[4] = arrival_delay
    x[AIRLINE_index] = 1
    x[ORIGIN_index] = 1
    x[DESTINATION_index] = 1
    x[DAY_OF_WEEK_index] = 1
    airlines_with_230_seats = ["AIRLINE_B6", "AIRLINE_AA", "AIRLINE_WN", "AIRLINE_EV", "AIRLINE_DL","AIRLINE_OO"]
    current_status = status_options1
    if current_status not in ["Weather", "Airline", "Nas", "Security"]:  # Assume 'None' or improper status goes to seating check
        # Check seat occupancy
        airlines_with_230_seats = ["AIRLINE_B6", "AIRLINE_AA", "AIRLINE_WN", "AIRLINE_EV", "AIRLINE_DL", "AIRLINE_OO"]
        total_seats = 230 if airline in airlines_with_230_seats else 350
        min_seats_required = total_seats / 2  # Minimum required seats

        if seats_occupied < min_seats_required:
            current_status = "Cancelled"
    
    prediction = reg_rf.predict([x])[0]
    return prediction, current_status

# def explain_prediction(model, features, prediction_result):
#     # Lấy cây đầu tiên từ mô hình Random Forest
#     tree = model.estimators_[0]
#     feature_importances = tree.tree_.compute_feature_importances(normalize=False)
#     importance_dict = dict(zip(features.columns, feature_importances))
    
#     # Sắp xếp các đặc điểm theo mức độ quan trọng và lấy ra 5 đặc điểm quan trọng nhất
#     important_features = sorted(importance_dict, key=importance_dict.get, reverse=True)[:5]
    
#     explanation = f"Dự đoán trì hoãn {abs(prediction_result):.2f} phút được ảnh hưởng chủ yếu bởi các yếu tố sau:\n"
#     for feature in important_features:
#         importance_score = importance_dict[feature]
#         detailed_explanation = ""
#         if feature == "ARRIVAL_DELAY":
#             detailed_explanation = "Thời gian trễ khi đến cao dẫn đến khả năng cao là \n chuyến bay tiếp theo cũng bị trì hoãn."
#         elif feature == "SCHEDULED_DEPARTURE":
#             detailed_explanation = "Thời điểm khởi hành trong ngày có thể ảnh hưởng lớn,\n với giờ cao điểm thường dẫn đến trì hoãn cao hơn."
#         elif feature == "DISTANCE":
#             detailed_explanation = "Chuyến bay dài hơn có thể yêu cầu thời gian chuẩn bị lâu hơn \n và có nhiều biến số về thời tiết và lưu lượng trên đường bay."
#         elif feature == "DAY":
#             detailed_explanation = "Ngày trong tháng có thể ảnh hưởng do lượng khách du lịch \n và các hoạt động khác tại sân bay."
#         elif feature == "MONTH":
#             detailed_explanation = "Mùa du lịch cao điểm như mùa hè hoặc các ngày lễ có thể làm tăng \n đáng kể khả năng và mức độ trì hoãn."
        
#         explanation += f"- {feature} ({importance_score:.4f}): {detailed_explanation}\n"
   
#     return explanation
def explain_prediction(model, features, prediction_result):
    # Lấy cây đầu tiên từ mô hình Random Forest
    tree = model.estimators_[0]
    feature_importances = tree.tree_.compute_feature_importances(normalize=False)
    importance_dict = dict(zip(features.columns, feature_importances))
    
    # Tính tổng độ quan trọng để chuẩn hóa
    total_importance = sum(importance_dict.values())

    # Sắp xếp các đặc điểm theo mức độ quan trọng và lấy ra 5 đặc điểm quan trọng nhất
    important_features = sorted(importance_dict, key=importance_dict.get, reverse=True)[:5]
    
    explanation = f"<div style='color: black;'>Dự đoán trì hoãn {abs(prediction_result):.2f} phút được ảnh hưởng chủ yếu bởi các yếu tố sau, chiếm tỷ lệ phần trăm của tổng độ quan trọng:<ul>"
    for feature in important_features:
        importance_score = importance_dict[feature]
        # Tính tỷ lệ phần trăm của độ quan trọng
        percentage_importance = (importance_score / total_importance) * 100
        detailed_explanation = ""
        if feature == "ARRIVAL_DELAY":
            detailed_explanation = "Thời gian trễ khi đến cao dẫn đến khả năng cao là chuyến bay tiếp theo cũng bị trì hoãn."
        elif feature == "SCHEDULED_DEPARTURE":
            detailed_explanation = "Thời điểm khởi hành trong ngày có thể ảnh hưởng lớn, với giờ cao điểm thường dẫn đến trì hoãn cao hơn."
        elif feature == "DISTANCE":
            detailed_explanation = "Chuyến bay dài hơn có thể yêu cầu thời gian chuẩn bị lâu hơn và có nhiều biến số về thời tiết và lưu lượng trên đường bay."
        elif feature == "DAY":
            detailed_explanation = "Ngày trong tháng có thể ảnh hưởng do lượng khách du lịch và các hoạt động khác tại sân bay."
        elif feature == "MONTH":
            detailed_explanation = "Mùa du lịch cao điểm như mùa hè hoặc các ngày lễ có thể làm tăng đáng kể khả năng và mức độ trì hoãn."
        
        explanation += f"<li>{feature} (Giá trị: {importance_score:.4f}, Chiếm: {percentage_importance:.2f}%): {detailed_explanation}</li>"
    explanation += "</ul></div>"
    return explanation


def display_result(prediction_result, current_status, explanation):
    
    message = ""
    explanation_displayed = False  # Biến để kiểm soát việc hiển thị giải thích

    if current_status == "Weather":
        message = "Thông báo: Theo chính sách của công ty, chuyến bay bị trì hoãn do điều kiện thời tiết xấu. Chúng tôi sẽ cập nhật thêm thông tin trong thời gian sớm nhất."
    elif current_status == "Airline":
        message = "Thông báo: Theo chính sách của công ty, chuyến bay bị trì hoãn do vấn đề kỹ thuật từ hãng hàng không. Chúng tôi đang nỗ lực khắc phục và sẽ thông báo lịch trình mới ngay khi có thể."
    elif current_status == "Nas":
        message = "Thông báo: Theo chính sách của công ty, chuyến bay bị trì hoãn do sự cố hệ thống hàng không quốc gia. Xin vui lòng chờ đợi thông báo tiếp theo từ chúng tôi."
    elif current_status == "Security":
        message = "Thông báo: Theo chính sách của công ty, chuyến bay bị trì hoãn do yêu cầu kiểm tra an ninh tăng cường. An toàn của hành khách và phi hành đoàn là ưu tiên hàng đầu của chúng tôi."
    elif current_status == "Cancelled":
        message = "Thông báo: Chuyến bay đã bị hủy do số lượng ghế đặt không đạt ngưỡng tối thiểu theo chính sách của công ty."
    else:
        if prediction_result >= 0:
            message = "Chuyến bay không bị trì hoãn. Sẽ khởi hành đúng giờ."
            explanation_displayed = True
        elif prediction_result >= -15:
            message = "Thông điệp này đang thông báo rằng chuyến bay chỉ bị trì hoãn " + str(abs(prediction_result)) + " phút. Một chuyến bay chỉ được coi là bị trì hoãn nếu thời gian chậm trễ lớn hơn 15 phút. Vì vậy, chuyến bay này không được xem là bị trì hoãn."
            explanation_displayed = True
        else:
            st.error(f"Thông điệp này đang thông báo rằng chuyến bay bị trì hoãn {abs(prediction_result)} phút. Một chuyến bay chỉ được coi là bị trì hoãn nếu thời gian chậm trễ lớn hơn 15 phút. Vì thời gian trì hoãn này vượt quá 15 phút, chuyến bay này được xem là bị trì hoãn.")
            explanation_displayed = True
     # Sử dụng markdown với HTML để thiết kế thông báo theo ý muốn
    st.markdown(f"<div style='color: black; padding: 10px; background-color: #f0f2f6; border-left: 5px solid #ffafbd;'>{message}</div>", unsafe_allow_html=True)
    if explanation_displayed:
        st.markdown("<h3 style='color: black;'>Giải thích Dự đoán:</h3>", unsafe_allow_html=True)
        st.markdown(explanation, unsafe_allow_html=True)
import joblib
import os
def main():
    
    st.markdown('<style>.main .block-container { width: 90%; margin-left: 5%; margin-right: auto; }</style>', unsafe_allow_html=True)
    st.markdown('<h1 style="color: blue;margin-left: 60px;">FLIGHT DELAY PREDICTION</h1>', unsafe_allow_html=True)
    set_bg("../32.jpg")
    month, day, sch_dept, distance, arrival_delay, airline, origin, destination, day_of_week,status_options1, seats_occupied = accept_data()
    if st.button("Load Data"):
        with st.spinner('Loading data... Please wait...'):
            final_data = loadData()
            X_train, y_train, X = preprocessing(final_data)
            reg_rf = rfg(X_train, y_train)
            st.session_state['model'] = reg_rf
            st.session_state['data'] = X
            st.success('Data loaded and model trained successfully!')

    if os.path.exists('model_random_forest.joblib') and st.button("Use Preloaded Data"):
        st.session_state['model'] = joblib.load('model_random_forest.joblib')
        st.session_state['data'] = pd.read_csv('final_data.csv')
        st.success("Preloaded data and model are now in use.")

    # if 'model' in st.session_state and st.button("Predict using Random Forest Regressor"):
    if 'model' in st.session_state:
        if st.button("Predict using Random Forest Regressor"):
            prediction_result, current_status = prediction(st.session_state['data'], month, day, sch_dept, distance, 
            arrival_delay, airline, origin, destination, day_of_week, status_options1, seats_occupied, st.session_state['model'])
            explanation = explain_prediction(st.session_state['model'], st.session_state['data'], prediction_result)
            display_result(prediction_result, current_status, explanation)
if __name__=='__main__':
    main()

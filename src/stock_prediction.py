import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
)
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 실행
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from supabase import create_client, Client

# 출력 디렉토리 생성
os.makedirs('outputs', exist_ok=True)

# GitHub Actions에서 환경변수로 받기
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')

if not url or not key:
    raise ValueError("SUPABASE_URL과 SUPABASE_KEY 환경변수가 설정되지 않았습니다.")

print(f"Connecting to Supabase: {url}")
supabase: Client = create_client(url, key)

def get_all_data(table_name):
    """Supabase에서 모든 데이터를 가져오는 함수"""
    all_data = []
    offset = 0
    limit = 1000  # Supabase의 기본 제한
    while True:
        response = supabase.table(table_name).select("*").order("날짜", desc=False).range(offset, offset + limit - 1).execute()
        data = response.data
        if not data:  # 더 이상 데이터가 없으면 종료
            break
        all_data.extend(data)
        offset += limit
    return all_data

def get_stock_data_from_db():
    """Supabase에서 주식 데이터를 가져오는 함수"""
    try:
        # 전체 데이터 가져오기
        all_data = get_all_data("economic_and_stock_data")
        print(f"economic_and_stock_data 테이블에서 {len(all_data)}개 데이터를 성공적으로 가져왔습니다!")
        df = pd.DataFrame(all_data)

        # 날짜 열을 datetime으로 변환하고 정렬
        df['날짜'] = pd.to_datetime(df['날짜'])
        df.sort_values(by='날짜', inplace=True)

        # 결측치 처리
        print("결측치 처리 중...")
        df = df.ffill().bfill()  # 앞/뒤 값으로 결측치 채우기

        # 수치형 컬럼으로 변환
        exclude_columns = ['날짜']
        numeric_columns = [col for col in df.columns if col not in exclude_columns]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # NaN 비율 확인
        nan_ratios = df[numeric_columns].isna().mean()
        print("수치형 컬럼별 NaN 비율:")
        print(nan_ratios)

        # 유효한 데이터가 있는 컬럼만 dropna 대상으로 설정
        valid_columns = [col for col in numeric_columns if nan_ratios[col] < 1.0]
        df.dropna(subset=valid_columns, inplace=True)

        print(f"처리 후 데이터 크기: {df.shape}")
        return df
    except Exception as e:
        print(f"데이터 가져오기 오류: {e}")
        return None

def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
    """Transformer Encoder 레이어"""
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    ffn = Dense(ff_dim, activation="relu")(attention_output)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn_output = Dropout(dropout)(ffn)
    ffn_output = Add()([attention_output, ffn_output])
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)

    return ffn_output

def build_transformer_with_two_inputs(stock_shape, econ_shape, num_heads, ff_dim, target_size):
    """두 개의 입력을 가진 Transformer 모델 구축"""
    stock_inputs = Input(shape=stock_shape)
    stock_encoded = stock_inputs
    for _ in range(4):  # 4개의 Transformer Layer
        stock_encoded = transformer_encoder(stock_encoded, num_heads=num_heads, ff_dim=ff_dim)
    stock_encoded = Dense(64, activation="relu")(stock_encoded)

    econ_inputs = Input(shape=econ_shape)
    econ_encoded = econ_inputs
    for _ in range(4):  # 4개의 Transformer Layer
        econ_encoded = transformer_encoder(econ_encoded, num_heads=num_heads, ff_dim=ff_dim)
    econ_encoded = Dense(64, activation="relu")(econ_encoded)

    merged = Add()([stock_encoded, econ_encoded])
    merged = Dense(128, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = GlobalAveragePooling1D()(merged)
    outputs = Dense(target_size)(merged)

    return Model(inputs=[stock_inputs, econ_inputs], outputs=outputs)

def save_predictions_to_db(result_df):
    """예측 결과를 Supabase에 저장하는 함수"""
    try:
        # 기존 테이블이 없으면 생성 (predicted_stocks 테이블에 저장)
        records = result_df.to_dict('records')

        # 테이블에 먼저 데이터 삭제 후 새로 삽입
        supabase.table("predicted_stocks").delete().neq("id", 0).execute()

        # 일괄 삽입 (큰 데이터라면 청크로 나누어 삽입)
        chunk_size = 100
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i+chunk_size]
            response = supabase.table("predicted_stocks").insert(chunk).execute()

        print(f"{len(records)}개의 예측 결과가 데이터베이스에 저장되었습니다.")
    except Exception as e:
        print(f"데이터베이스 저장 오류: {e}")

def get_predictions_from_db(chunk_size=1000):
    """Supabase에서 예측 데이터 가져오기"""
    try:
        # 전체 레코드 수 확인
        count_response = supabase.table("predicted_stocks").select("id", count="exact").execute()
        total_count = count_response.count
        print(f"predicted_stocks 테이블의 총 레코드 수: {total_count}")

        # 데이터를 저장할 빈 리스트
        all_data = []

        # 청크 단위로 데이터 가져오기
        for offset in range(0, total_count, chunk_size):
            response = (
                supabase.table("predicted_stocks")
                .select("*")
                .order("날짜", desc=False)
                .range(offset, offset + chunk_size - 1)
                .execute()
            )
            chunk_data = response.data
            print(f"오프셋 {offset}에서 {len(chunk_data)}개 데이터를 가져왔습니다.")
            all_data.extend(chunk_data)

        # 모든 데이터를 DataFrame으로 변환
        df = pd.DataFrame(all_data)
        print(f"총 {len(df)}개 데이터를 성공적으로 가져왔습니다!")

        # 날짜 열을 datetime으로 변환
        df['날짜'] = pd.to_datetime(df['날짜'])

        return df
    except Exception as e:
        print(f"데이터 가져오기 오류: {e}")
        return None

def save_analysis_to_db(result_df):
    """분석 결과를 Supabase에 저장하는 함수"""
    try:
        # stock_analysis_results 테이블에 저장
        records = result_df.to_dict('records')

        # 테이블에 먼저 데이터 삭제 후 새로 삽입
        supabase.table("stock_analysis_results").delete().neq("id", 0).execute()

        # 일괄 삽입 (큰 데이터라면 청크로 나누어 삽입)
        chunk_size = 100
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i+chunk_size]
            response = supabase.table("stock_analysis_results").insert(chunk).execute()

        print(f"{len(records)}개의 분석 결과가 데이터베이스에 저장되었습니다.")
    except Exception as e:
        print(f"데이터베이스 저장 오류: {e}")

def evaluate_predictions(data, target_columns, forecast_horizon):
    """예측 성능을 평가하는 함수"""
    metrics = []

    for col in target_columns:
        predicted_col = f'{col}_Predicted'
        actual_col = f'{col}_Actual'

        # Check if the columns exist
        if predicted_col not in data.columns or actual_col not in data.columns:
            print(f"Skipping {col}: Columns not found in data ({predicted_col}, {actual_col})")
            continue

        # Retrieve predicted and actual values
        predicted = data[predicted_col]
        # Shift the actual values by forecast_horizon days
        actual = data[actual_col].shift(-forecast_horizon)

        # Use only valid (non-NaN) indices
        valid_idx = ~predicted.isna() & ~actual.isna()
        predicted = predicted[valid_idx]
        actual = actual[valid_idx]

        if len(predicted) == 0:
            print(f"Skipping {col}: No valid prediction/actual pairs.")
            continue

        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = mse ** 0.5
        mape = (abs((actual - predicted) / actual).mean()) * 100
        accuracy = 100 - mape

        metrics.append({
            'Stock': col,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'Accuracy (%)': accuracy
        })

    return pd.DataFrame(metrics)

def analyze_rise_predictions(data, target_columns):
    """상승/하락 예측을 분석하는 함수"""
    last_row = data.iloc[-1]
    results = []

    for col in target_columns:
        actual_col = f'{col}_Actual'
        predicted_col = f'{col}_Predicted'

        last_actual_price = last_row.get(actual_col, np.nan)
        predicted_future_price = last_row.get(predicted_col, np.nan)

        # Determine rise/fall and rise percentage
        if pd.notna(last_actual_price) and pd.notna(predicted_future_price):
            predicted_rise = predicted_future_price > last_actual_price
            rise_probability = ((predicted_future_price - last_actual_price) / last_actual_price) * 100
        else:
            predicted_rise = np.nan
            rise_probability = np.nan

        results.append({
            'Stock': col,
            'Last Actual Price': last_actual_price,
            'Predicted Future Price': predicted_future_price,
            'Predicted Rise': predicted_rise,
            'Rise Probability (%)': rise_probability
        })

    return pd.DataFrame(results)

def generate_recommendation(row):
    """매수/매도 추천을 생성하는 함수"""
    rise_prob = row.get('Rise Probability (%)', 0)
    predicted_rise = row.get('Predicted Rise', False)

    if pd.isna(rise_prob) or pd.isna(predicted_rise):
        return "No Data"

    if predicted_rise and rise_prob > 0:
        if rise_prob > 2:
            return "STRONG BUY"
        else:
            return "BUY"
    else:
        return "SELL"

def generate_analysis(row):
    """분석 코멘트를 생성하는 함수"""
    stock_name = row['Stock']
    rise_prob = row.get('Rise Probability (%)', 0)
    predicted_rise = row.get('Predicted Rise', False)

    if pd.isna(rise_prob) or pd.isna(predicted_rise):
        return f"{stock_name}: Not enough data"

    if predicted_rise:
        return f"{stock_name} is expected to rise by about {rise_prob:.2f}%. Consider buying or holding."
    else:
        return f"{stock_name} is expected to fall by about {-rise_prob:.2f}%. A cautious approach is recommended."

def main():
    """메인 실행 함수"""
    try:
        print("Starting stock prediction process...")
        
        # 대상 컬럼들 정의
        target_columns = [
            '애플', '마이크로소프트', '아마존', '구글 A', '구글 C', '메타',
            '테슬라', '엔비디아', '코스트코', '넷플릭스', '페이팔', '인텔', '시스코', '컴캐스트',
            '펩시코', '암젠', '허니웰 인터내셔널', '스타벅스', '몬델리즈', '마이크론', '브로드컴',
            '어도비', '텍사스 인스트루먼트', 'AMD', '어플라이드 머티리얼즈', 'S&P 500 ETF', 'QQQ ETF'
        ]

        economic_features = [
            '10년 기대 인플레이션율', '장단기 금리차', '기준금리', '미시간대 소비자 심리지수',
            '실업률', '2년 만기 미국 국채 수익률', '10년 만기 미국 국채 수익률', '금융스트레스지수',
            '개인 소비 지출', '소비자 물가지수', '5년 변동금리 모기지', '미국 달러 환율',
            '통화 공급량 M2', '가계 부채 비율', 'GDP 성장률', '나스닥 종합지수', 'S&P 500 지수', '금 가격', '달러 인덱스', '나스닥 100',
            'S&P 500 ETF', 'QQQ ETF', '러셀 2000 ETF', '다우 존스 ETF', 'VIX 지수',
            '닛케이 225', '상해종합', '항셍', '영국 FTSE', '독일 DAX', '프랑스 CAC 40',
            '미국 전체 채권시장 ETF', 'TIPS ETF', '투자등급 회사채 ETF', '달러/엔', '달러/위안',
            '미국 리츠 ETF'
        ]

        forecast_horizon = 14  # 예측 기간 (14일 후를 예측)
        lookback = 90

        # 데이터 로드
        print("Loading data from database...")
        data = get_stock_data_from_db()
        if data is None or data.empty:
            raise ValueError("DB에서 데이터를 가져오지 못했습니다. 테이블과 컬럼명을 확인하세요.")

        print("Scaling data...")
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        data_scaled = data.copy()
        stock_scaler = MinMaxScaler()
        econ_scaler = MinMaxScaler()

        data_scaled[target_columns] = stock_scaler.fit_transform(data[target_columns])
        data_scaled[economic_features] = econ_scaler.fit_transform(data[economic_features])

        # 훈련 데이터 생성
        X_stock_train = []
        X_econ_train = []
        y_train = []

        for i in range(lookback, len(data_scaled) - forecast_horizon):
            X_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy()
            X_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy()
            y_val = data_scaled[target_columns].iloc[i + forecast_horizon - 1].to_numpy()
            X_stock_train.append(X_stock_seq)
            X_econ_train.append(X_econ_seq)
            y_train.append(y_val)

        X_stock_train = np.array(X_stock_train)
        X_econ_train = np.array(X_econ_train)
        y_train = np.array(y_train)

        # 전체 예측 데이터 생성
        X_stock_full = []
        X_econ_full = []
        for i in range(lookback, len(data_scaled)):
            X_stock_seq = data_scaled[target_columns].iloc[i - lookback:i].to_numpy()
            X_econ_seq = data_scaled[economic_features].iloc[i - lookback:i].to_numpy()
            X_stock_full.append(X_stock_seq)
            X_econ_full.append(X_econ_seq)

        X_stock_full = np.array(X_stock_full)
        X_econ_full = np.array(X_econ_full)

        print("Building Transformer model...")
        stock_shape = (lookback, len(target_columns))
        econ_shape = (lookback, len(economic_features))

        model = build_transformer_with_two_inputs(stock_shape, econ_shape, num_heads=8, ff_dim=256, target_size=len(target_columns))
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
        model.summary()

        print("Training model...")
        history = model.fit([X_stock_train, X_econ_train], y_train, epochs=50, batch_size=32, verbose=1)

        print("Performing full predictions...")
        predicted_prices = model.predict([X_stock_full, X_econ_full], verbose=1)
        predicted_prices_actual = stock_scaler.inverse_transform(predicted_prices)

        pred_len = len(predicted_prices_actual)

        # 오늘 날짜들 (마지막 날짜까지 포함)
        today_dates = data['날짜'].iloc[lookback : lookback + pred_len].values

        # 오늘 실제 주가
        actual_data_end = min(lookback + pred_len, len(data))
        actual_full = data[target_columns].iloc[lookback:actual_data_end].values

        # 만약 actual_full 길이가 pred_len보다 짧다면 부족한 부분을 NaN으로 채움
        if actual_full.shape[0] < pred_len:
            nan_padding = np.full((pred_len - actual_full.shape[0], len(target_columns)), np.nan)
            actual_full = np.vstack([actual_full, nan_padding])

        result_data = pd.DataFrame({'날짜': today_dates})

        for idx, col in enumerate(target_columns):
            result_data[f'{col}_Predicted'] = predicted_prices_actual[:, idx]
            result_data[f'{col}_Actual'] = actual_full[:, idx]

        result_data['날짜'] = pd.to_datetime(result_data['날짜'], errors='coerce')
        result_data['날짜'] = result_data['날짜'].dt.strftime('%Y-%m-%d')

        # 예측 결과 저장
        save_predictions_to_db(result_data)

        # 훈련 손실 그래프 저장
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('outputs/training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 각 주식별 예측 그래프 저장 (상위 5개만)
        for i, col in enumerate(target_columns[:5]):  # 너무 많은 그래프 생성 방지
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(result_data['날짜']), result_data[f'{col}_Actual'], label='Actual (Today)', alpha=0.7)
            plt.plot(pd.to_datetime(result_data['날짜']), result_data[f'{col}_Predicted'], label=f'Predicted ({forecast_horizon} days later)', alpha=0.7)
            plt.title(f'{col} - Actual(Today) vs Predicted({forecast_horizon} days later)')
            plt.xlabel('Date (Today)')
            plt.ylabel('Price')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            plt.savefig(f'outputs/{col.replace("/", "_")}_prediction.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("================================")
        print("결과 추론 시작")
        print("================================")

        # 분석 실행
        print("Starting analysis...")
        analysis_data = get_predictions_from_db(chunk_size=1000)
        if analysis_data is not None and len(analysis_data) > 0:
            # 평가 수행
            evaluation_results = evaluate_predictions(analysis_data, target_columns, forecast_horizon)
            print("============ Evaluation Results ============")
            print(evaluation_results)

            # 상승 예측 분석
            rise_results = analyze_rise_predictions(analysis_data, target_columns)
            print("============ Rise Predictions ============")
            print(rise_results)

            # 결과 병합
            final_results = pd.merge(evaluation_results, rise_results, on='Stock', how='outer')
            final_results = final_results.sort_values(by='Rise Probability (%)', ascending=False)
            final_results['Recommendation'] = final_results.apply(generate_recommendation, axis=1)
            final_results['Analysis'] = final_results.apply(generate_analysis, axis=1)

            # 컬럼 순서 정리
            column_order = [
                'Stock',
                'MAE', 'MSE', 'RMSE', 'MAPE (%)', 'Accuracy (%)',
                'Last Actual Price', 'Predicted Future Price', 'Predicted Rise', 'Rise Probability (%)',
                'Recommendation', 'Analysis'
            ]
            final_results = final_results[column_order]

            # 분석 결과 저장
            save_analysis_to_db(final_results)

            # 결과를 CSV로도 저장
            final_results.to_csv('outputs/analysis_results.csv', index=False)

            print("=============== Final Report ===============")
            print(final_results.to_string(index=False))

            print("\n분석 결과가 'stock_analysis_results' 테이블에 저장되었습니다.")
        
        print("Stock prediction and analysis completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()
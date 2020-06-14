import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
# tensorflow ver 1.13.1
import datetime
# numpy ver 1.16.2
import numpy as np

# dataframe 출력 세팅
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

# 종목이름으로 종목 code를 리턴하는 함수
def get_stockCode(name, df) :
    for i in range(len(df)) :
        if df.iloc[i]['회사명'] == name :
            code = df.iloc[i]['종목코드']
            break
    url = "http://finance.naver.com/item/sise_day.nhn?code=" + code
    return url

# minmax 정규화 함수
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 0으로 나누는 오류 예방차원

# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

def predictor(stock_df, inputSec) :
    test_df = pd.DataFrame()
    today_pri = 0
    # 종목을 입력받으면 코드를 통해서 최근 3년간의 주가를 naver finance에서 가져옴
    url = get_stockCode(inputSec, stock_df)
    for page in range(1, 75):
        pg_url = '{url}&page={page}'.format(url=url, page=page)
        test_df = test_df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
    test_df = test_df.dropna()

    # 해당 종목 주가의 데이터 확인
    print(test_df.head())
    print(test_df.tail())
    print(test_df.info())

    # 오늘의 종가 기록
    today_pri = test_df.iloc[0]['종가']

    # 데이터 전처리
    test_df = test_df.rename(
        columns={'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low',
                 '거래량': 'volume'})

    # int타입으로 데이터 변환
    test_df[['close', 'diff', 'open', 'high', 'low', 'volume']] \
        = test_df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)

    # 일자(date)를 기준으로 오름차순 정렬
    test_df = test_df.sort_values(by=['date'], ascending=True)

    # 랜덤시드 설정
    tf.set_random_seed(7)

    # def data_standardization(x):
    #    x_np = np.asarray(x)
    #    return (x_np - x_np.mean()) / x_np.std()

    # 하이퍼파라미터
    input_data_column_cnt = 6  # 입력데이터의 컬럼 개수(Variable 개수)
    output_data_column_cnt = 1  # 결과데이터의 컬럼 개수

    seq_length = 28  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
    rnn_cell_hidden_dim = 20  # 각 셀의 (hidden)출력 크기
    forget_bias = 1.0  # 망각편향(기본값 1.0)
    num_stacked_layers = 1  # stacked LSTM layers 개수
    keep_prob = 1  # dropout할 때 keep할 비율

    epoch_num = 1000  # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
    learning_rate = 0.01  # 학습률

    raw_dataframe = test_df.copy()

    # 시간열을 제거하고 dataframe 재생성하지 않기
    raw_dataframe.drop('date', axis=1, inplace=True)

    stock_info = raw_dataframe.values[1:].astype(np.float)  # 금액&거래량 문자열을 부동소수점형으로 변환한다
    print("stock_info.shape: ", stock_info.shape)
    print("stock_info[0]: ", stock_info[0])

    # 데이터 전처리
    # 가격과 거래량 수치의 차이가 많아나서 각각 별도로 정규화한다

    # 가격형태 데이터들을 정규화한다
    # ['close','diff','open','high','low','Volume']에서 'low'까지 취함
    # 곧, 마지막 열 Volume를 제외한 모든 열
    price = stock_info[:, :-1]
    norm_price = min_max_scaling(price)  # 가격형태 데이터 정규화 처리
    print("price.shape: ", price.shape)
    print("price[0]: ", price[0])
    print("norm_price[0]: ", norm_price[0])
    print("=" * 100)  # 화면상 구분용

    # 거래량형태 데이터를 정규화한다
    # ['close','diff','open','high','low','Volume']에서 마지막 'Volume'만 취함
    # [:,-1]이 아닌 [:,-1:]이므로 주의하자! 스칼라가아닌 벡터값 산출해야만 쉽게 병합 가능
    volume = stock_info[:, -1:]
    norm_volume = min_max_scaling(volume)  # 거래량형태 데이터 정규화 처리
    print("volume.shape: ", volume.shape)
    print("volume[0]: ", volume[0])
    print("norm_volume[0]: ", norm_volume[0])
    print("=" * 100)  # 화면상 구분용

    # 행은 그대로 두고 열을 우측에 붙여 합친다
    x = np.concatenate((norm_price, norm_volume), axis=1)  # axis=1, 세로로 합친다
    print("x.shape: ", x.shape)
    print("x[0]: ", x[0])  # x의 첫 값
    print("x[-1]: ", x[-1])  # x의 마지막 값
    print("=" * 100)  # 화면상 구분용

    y = x[:, [0]]  # 타켓은 주식 종가이다
    print("y[0]: ", y[0])  # y의 첫 값
    print("y[-1]: ", y[-1])  # y의 마지막 값

    dataX = []  # 입력으로 사용될 Sequence Data
    dataY = []  # 출력(타켓)으로 사용

    for i in range(0, len(y) - seq_length):
        _x = x[i: i + seq_length]
        _y = y[i + seq_length]  # 다음 나타날 주가(정답)
        if i is 0:
            print(_x, "->", _y)  # 첫번째 행만 출력해 봄
        dataX.append(_x)  # dataX 리스트에 추가
        dataY.append(_y)  # dataY 리스트에 추가

    # 학습용/테스트용 데이터 생성
    # 전체 70%를 학습용 데이터로 사용
    train_size = int(len(dataY) * 0.7)
    # 나머지(30%)를 테스트용 데이터로 사용
    test_size = len(dataY) - train_size

    # 데이터를 잘라 학습용 데이터 생성
    trainX = np.array(dataX[0:train_size])
    trainY = np.array(dataY[0:train_size])

    # 데이터를 잘라 테스트용 데이터 생성
    testX = np.array(dataX[train_size:len(dataX)])
    testY = np.array(dataY[train_size:len(dataY)])

    # 텐서플로우 플레이스홀더 생성
    # 입력 X, 출력 Y를 생성한다
    X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
    print("X: ", X)
    Y = tf.placeholder(tf.float32, [None, output_data_column_cnt])
    print("Y: ", Y)

    # 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
    targets = tf.placeholder(tf.float32, [None, 1])
    print("targets: ", targets)

    predictions = tf.placeholder(tf.float32, [None, 1])
    print("predictions: ", predictions)

    # 모델(LSTM 네트워크) 생성
    def lstm_cell():
        # LSTM셀을 생성
        # num_units: 각 Cell 출력 크기
        # forget_bias:  to the biases of the forget gate
        # (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
        # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
        # state_is_tuple: False ==> they are concatenated along the column axis.
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,
                                            forget_bias=forget_bias, state_is_tuple=True,
                                            activation=tf.nn.tanh)
        if keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return cell

    # num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
    stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
    multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs,
                                              state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

    # RNN Cell(여기서는 LSTM셀임)들을 연결
    hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
    print("hypothesis: ", hypothesis)

    # [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
    # 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기때문에 MANY-TO-ONE형태이다
    hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt,
                                                   activation_fn=tf.identity)

    # 손실함수로 평균제곱오차를 사용한다
    loss = tf.reduce_sum(tf.square(hypothesis - Y))
    # 최적화함수로 AdamOptimizer를 사용한다
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train = optimizer.minimize(loss)

    # RMSE(Root Mean Square Error)
    # 제곱오차의 평균을 구하고 다시 제곱근을 구하면 평균 오차가 나온다
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

    train_error_summary = []  # 학습용 데이터의 오류를 중간 중간 기록한다
    test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다
    test_predict = ''  # 테스트용데이터로 예측한 결과

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 학습한다
    start_time = datetime.datetime.now()  # 시작시간을 기록한다
    print('학습을 시작합니다...')
    for epoch in range(epoch_num):
        _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):  # 100번째마다 또는 마지막 epoch인 경우
            # 학습용데이터로 rmse오차를 구한다
            train_predict = sess.run(hypothesis, feed_dict={X: trainX})
            train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
            train_error_summary.append(train_error)

            # 테스트용데이터로 rmse오차를 구한다
            test_predict = sess.run(hypothesis, feed_dict={X: testX})
            test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
            test_error_summary.append(test_error)

            # 현재 오류를 출력한다
            print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch + 1, train_error,
                                                                                     test_error,
                                                                                     test_error - train_error))

    end_time = datetime.datetime.now()  # 종료시간을 기록한다
    elapsed_time = end_time - start_time  # 경과시간을 구한다
    print('elapsed_time:', elapsed_time)
    print('elapsed_time per epoch:', elapsed_time / epoch_num)

    # 결과 그래프 출력
    plt.figure(1)
    # 노란 그래프가 학습 데이터의 학습율
    # 파란 그래프가 실제 데이터의 학습율
    plt.plot(train_error_summary, 'gold')
    plt.plot(test_error_summary, 'b')
    plt.xlabel('Epoch(x100)')
    plt.ylabel('Root Mean Square Error')

    plt.figure(2)
    plt.plot(testY, 'r')
    plt.plot(test_predict, 'b')
    plt.xlabel('Time Period')
    plt.ylabel('Stock Price')
    plt.show()

    # sequence length만큼의 가장 최근 데이터를 슬라이싱한다
    recent_data = np.array([x[len(x) - seq_length:]])
    print("recent_data.shape:", recent_data.shape)
    print("recent_data:", recent_data)

    # 내일 종가를 예측해본다
    test_predict = sess.run(hypothesis, feed_dict={X: recent_data})

    print("test_predict", test_predict[0])
    test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터 역정규화한다
    print("Today's stock price of ", inputSec, today_pri)
    print("Tomorrow's stock price", test_predict[0])  # 예측한 주가를 출력한다
    print("전일비 : ", test_predict[0] / today_pri * 100, "%")

    tf.reset_default_graph()


def multiPredictor(stock_df) :
    ranking = pd.DataFrame(columns=["종목", "현재가격", "내일가격", "전일비"])
    test_df = pd.DataFrame()
    today_pri = 0
    for stock in range(len(stock_df)) :
        test_df = pd.DataFrame()
        url = get_stockCode(stock_df.iloc[stock][0], stock_df)
        for page in range(1, 75):
            pg_url = '{url}&page={page}'.format(url=url, page=page)
            test_df = test_df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
        test_df = test_df.dropna()

        # 데이터 전처리
        test_df = test_df.rename(
            columns={'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low',
                     '거래량': 'volume'})

        # int타입으로 데이터 변환
        test_df[['close', 'diff', 'open', 'high', 'low', 'volume']] \
            = test_df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)

        # 오늘의 종가 기록
        today_pri = test_df.iloc[0]["close"]

        # 일자(date)를 기준으로 오름차순 정렬
        test_df = test_df.sort_values(by=['date'], ascending=True)

        # 랜덤시드 설정
        tf.set_random_seed(7)

        # def data_standardization(x):
        #    x_np = np.asarray(x)
        #    return (x_np - x_np.mean()) / x_np.std()

        # 하이퍼파라미터
        input_data_column_cnt = 6  # 입력데이터의 컬럼 개수(Variable 개수)
        output_data_column_cnt = 1  # 결과데이터의 컬럼 개수

        seq_length = 28  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
        rnn_cell_hidden_dim = 20  # 각 셀의 (hidden)출력 크기
        forget_bias = 1.0  # 망각편향(기본값 1.0)
        num_stacked_layers = 1  # stacked LSTM layers 개수
        keep_prob = 1  # dropout할 때 keep할 비율

        epoch_num = 1000  # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
        learning_rate = 0.01  # 학습률

        raw_dataframe = test_df.copy()

        # 시간열을 제거하고 dataframe 재생성하지 않기
        raw_dataframe.drop('date', axis=1, inplace=True)

        stock_info = raw_dataframe.values[1:].astype(np.float)  # 금액&거래량 문자열을 부동소수점형으로 변환한다

        # 데이터 전처리
        # 가격과 거래량 수치의 차이가 많아나서 각각 별도로 정규화한다

        # 가격형태 데이터들을 정규화한다
        # ['close','diff','open','high','low','Volume']에서 'low'까지 취함
        # 곧, 마지막 열 Volume를 제외한 모든 열
        price = stock_info[:, :-1]
        norm_price = min_max_scaling(price)  # 가격형태 데이터 정규화 처리

        # 거래량형태 데이터를 정규화한다
        # ['close','diff','open','high','low','Volume']에서 마지막 'Volume'만 취함
        # [:,-1]이 아닌 [:,-1:]이므로 주의하자! 스칼라가아닌 벡터값 산출해야만 쉽게 병합 가능
        volume = stock_info[:, -1:]
        norm_volume = min_max_scaling(volume)  # 거래량형태 데이터 정규화 처리

        # 행은 그대로 두고 열을 우측에 붙여 합친다
        x = np.concatenate((norm_price, norm_volume), axis=1)  # axis=1, 세로로 합친다
        y = x[:, [0]]  # 타켓은 주식 종가이다

        dataX = []  # 입력으로 사용될 Sequence Data
        dataY = []  # 출력(타켓)으로 사용

        for i in range(0, len(y) - seq_length):
            _x = x[i: i + seq_length]
            _y = y[i + seq_length]  # 다음 나타날 주가(정답)
            dataX.append(_x)  # dataX 리스트에 추가
            dataY.append(_y)  # dataY 리스트에 추가

        # 학습용/테스트용 데이터 생성
        # 전체 70%를 학습용 데이터로 사용
        train_size = int(len(dataY) * 0.7)
        # 나머지(30%)를 테스트용 데이터로 사용
        test_size = len(dataY) - train_size

        # 데이터를 잘라 학습용 데이터 생성
        trainX = np.array(dataX[0:train_size])
        trainY = np.array(dataY[0:train_size])

        # 데이터를 잘라 테스트용 데이터 생성
        testX = np.array(dataX[train_size:len(dataX)])
        testY = np.array(dataY[train_size:len(dataY)])

        # 텐서플로우 플레이스홀더 생성
        # 입력 X, 출력 Y를 생성한다
        X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
        Y = tf.placeholder(tf.float32, [None, output_data_column_cnt])

        # 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
        targets = tf.placeholder(tf.float32, [None, 1])

        predictions = tf.placeholder(tf.float32, [None, 1])

        # 모델(LSTM 네트워크) 생성
        def lstm_cell():
            # LSTM셀을 생성
            # num_units: 각 Cell 출력 크기
            # forget_bias:  to the biases of the forget gate
            # (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
            # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
            # state_is_tuple: False ==> they are concatenated along the column axis.
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,
                                                forget_bias=forget_bias, state_is_tuple=True,
                                                activation=tf.nn.tanh)
            if keep_prob < 1.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        # num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
        stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
        multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs,
                                                  state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

        # RNN Cell(여기서는 LSTM셀임)들을 연결
        hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

        # [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
        # 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기때문에 MANY-TO-ONE형태이다
        hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt,
                                                       activation_fn=tf.identity)

        # 손실함수로 평균제곱오차를 사용한다
        loss = tf.reduce_sum(tf.square(hypothesis - Y))
        # 최적화함수로 AdamOptimizer를 사용한다
        optimizer = tf.train.AdamOptimizer(learning_rate)

        train = optimizer.minimize(loss)

        # RMSE(Root Mean Square Error)
        # 제곱오차의 평균을 구하고 다시 제곱근을 구하면 평균 오차가 나온다
        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

        train_error_summary = []  # 학습용 데이터의 오류를 중간 중간 기록한다
        test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다
        test_predict = ''  # 테스트용데이터로 예측한 결과

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # 학습한다
        start_time = datetime.datetime.now()  # 시작시간을 기록한다
        print(stock, '번째 학습을 시작합니다...')
        for epoch in range(epoch_num):
            _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
            if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):  # 100번째마다 또는 마지막 epoch인 경우
                # 학습용데이터로 rmse오차를 구한다
                train_predict = sess.run(hypothesis, feed_dict={X: trainX})
                train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
                train_error_summary.append(train_error)

                # 테스트용데이터로 rmse오차를 구한다
                test_predict = sess.run(hypothesis, feed_dict={X: testX})
                test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
                test_error_summary.append(test_error)

        # sequence length만큼의 가장 최근 데이터를 슬라이싱한다
        recent_data = np.array([x[len(x) - seq_length:]])

        # 내일 종가를 예측해본다
        test_predict = sess.run(hypothesis, feed_dict={X: recent_data})
        test_predict = reverse_min_max_scaling(price, test_predict)  # 금액데이터 역정규화
        test_diff = test_predict[0] / today_pri * 100
        ranking.loc[stock] = [stock_df.iloc[stock][0], today_pri, test_predict[0], test_diff]
        tf.reset_default_graph()
        print(stock_df.iloc[stock][0], today_pri, test_predict[0], test_diff)
    print(ranking.sort_values(by=["전일비"]))


def run() :
    # stock 데이터 전처리
    STOCK_RANK = 100
    try :
        # 미리 전처리된 주가 데이터가 있다면 그 데이터를 사용합니다.
        stock_df = pd.read_csv("./stock_df.csv")
        stock_df.drop("Unnamed: 0", axis=1, inplace=True)

    except :
        # 주식정보와 주가데이터를 이용해서 주식정보에 최근일 거래량을 달아즙니다.
        stock_df = \
        pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0]
        stock_df = stock_df[["회사명", "종목코드", "업종", "상장일"]]
        stock_df['상장일'] = pd.to_datetime(stock_df['상장일'])
        stock_df.종목코드 = stock_df.종목코드.map("{:06d}".format)
        volumeArray = []
        for stock in range(len(stock_df)):
            test_df = pd.DataFrame()
            url = get_stockCode(stock_df.iloc[stock][0], stock_df)
            page = 1
            pg_url = '{url}&page={page}'.format(url=url, page=page)
            test_df = test_df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
            test_df = test_df.dropna()

            test_df.drop(['날짜', '종가', '전일비', '시가', '고가', '저가'], axis=1, inplace=True)
            volumeArray.append(test_df.iloc[0][0])
            if stock % 100 == 0 and stock != 0:
                print(stock / len(stock_df) * 100, "% 로딩중입니다...")
        stock_df["최근일 거래량"] = volumeArray

        stock_df.to_csv("./stock_df.csv")

    print("로딩완료!")
    print("=" * 100)

    while 1 :
        print("단타용 주가 예측 및 종목 추천 프로그램입니다.")
        print("1. 프로그램 설명")
        print("2. 모든 종목 조회")
        print("3. 단일 종목 단타 예측")
        print("4. 종목 추천")
        inputFir = input()

        if inputFir == "1" :
            print("Naver Finance의 데이터를 이용해 분석을 하는 프로그램입니다.")
            print("2017 ~ 현재 사이에 액면분할을 한 주식은 이슈가 존재합니다 ㅠ")
            print("상장일이 3년 미만인 주식은 표현되지 않습니다.")

        elif inputFir == "2" :
            print(stock_df)
            print(stock_df.info())

        elif inputFir == "3" :
            try :
                inputSec = input("종목을 입력해주세요.")
                predictor(stock_df, inputSec)
            except :
                print("다시 입력해주세요!")

        elif inputFir == "4" :
            stock_df.sort_values(by=['최근일 거래량'], inplace=True, ascending=False)
            for stock in range(len(stock_df)):
                if stock >= STOCK_RANK:
                    stock_df.drop(stock, inplace=True)
            multiPredictor(stock_df)

        else : print("다시 입력해주세요!")


run()
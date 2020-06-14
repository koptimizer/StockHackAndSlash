# StockHackAndSlash
- 매일 국내주식의 단타용 종목을 추천해주는 RNN기반 추천시스템입니다.

## 왜 만들었죠?
  <p align = 'center'>
    <img src = "https://github.com/koptimizer/StockHackAndSlash/blob/master/pics/booms.jpg" height= "500px" ><br>
  </p>
  
- 모의투자에서 최하위권을 달성했거든요...ㅠㅠ
- <b>"미래의 주가를 예측해서 떡상 할 주식을 알 수 있다면 얼마나 좋을까?"</b>에서 출발했습니다.
- 다만 주가를 예측하는 것은 어려운 일이라 몇가지 타협을 했습니다.
  - 장기적인 예측은 매우 어려우므로 내일의 주가를 예측하는 단타용 모델을 만들자
  - 모든 주식에 대해 모델을 돌리는 것은 너무 무거워서 전일비와 거래량에 따른 군집화를 시행하고 그 군집에 해당하는 주식들의 내일 주가를 예측하고 내림차순으로 서비스하는 앱을 생각했습니다.
  - 주가 데이터는 naver finance에서 가져왔고, LSTM이 시계열에 강건하기 때문에 채택했습니다.

## 기능
- <b>주식 정보 조회</b>
  - 국내에 상장된 모든 주식의 간단한 정보를 볼 수 있습니다.
- <b>단일 종목 단타 예측</b>
  - 회사명을 입력하면 내일의 주가를 예측해줍니다.
- <b>단타용 종목 추천</b>
  - 거래량 상위 100개의 주식을 선택합니다.
  - 100개의 주식의 내일 가격을 예측하고, 순서대로 정렬해서 종목을 추천합니다.
  
## 성능 평가
<p align = 'center'>
    <img src = "https://github.com/koptimizer/StockHackAndSlash/blob/master/pics/per.jpg" ><br>
  <img src = "https://github.com/koptimizer/StockHackAndSlash/blob/master/pics/%EB%96%A1%EC%83%81%EA%B0%80%EC%9E%90.JPG" ><br>
  </p>

## Environment & License
- ```Python 3.x```
- ```tensorflow ver 1.13.1```
- ```numpy ver 1.16.2```
- ```MIT License```

## 이슈
- 회사나 정책에 대한 뉴스, 성과에 대한 정보, 각종 지수등도 크롤링해서 활용한다면 더 높은 성능을 확보할 수 있을듯 합니다.
- 액면분할 및 거래정지, 떡락의 이슈가 존재하는 종목에 취약합니다.


# StockHackAndSlash
- 매일 국내주식의 단타용 종목을 추천해주는 LSTM기반 추천시스템입니다.
- Naver finance의 주식정보 및 주가 데이터를 이용했습니다.


## Environment & License
- ```Python 3.x```
- ```tensorflow ver 1.13.1```
- ```numpy ver 1.16.2```
- ```MIT License```

## 기능
- 주식 정보 조회
  - 국내에 상장된 모든 주식의 간단한 정보를 볼 수 있습니다.
- 단일 종목 단타 예측
  - 회사명을 입력하면 내일의 주가를 예측해줍니다.
- 단타용 종목 추천
  - 거래량 상위 100개의 주식을 선택합니다.
  - 100개의 주식의 내일 가격을 예측하고, 순서대로 정렬해서 종목을 추천합니다.
  
## 성능 평가
<p align = 'center'>
    <img src = "https://github.com/koptimizer/StockHackAndSlash/blob/master/pics/per.jpg" ><br>
  <img src = "https://github.com/koptimizer/StockHackAndSlash/blob/master/pics/%EB%96%A1%EC%83%81%EA%B0%80%EC%9E%90.JPG" ><br>
  </p>

  
## 이슈
- 회사나 정책에 대한 뉴스, 성과에 대한 정보, 각종 지수등도 크롤링해서 활용한다면 더 높은 성능을 확보할 수 있을듯 합니다.
- 액면분할 및 거래정지, 떡락의 이슈가 존재하는 종목에 취약합니다.


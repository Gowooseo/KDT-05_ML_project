##--------------------------------------------------------------
## 모델을 활용한 서비스 제공
##--------------------------------------------------------------
# 모듈 로딩
from joblib import load

#전역 변수
model_file='../model/bald_dt.pkl'

#모델 로딩
model=load(model_file)

# 로딩된 모델 확인
print(model.classes_)

#붓꽃 정보 입력 => 4개 피쳐
datas=input("피쳐 정보 입력 (예:'나이', '스트레스 수치(0~10)', '결혼 여부(0(하지 않음) or 1(함))', '흡연 여부(0(없음) or 1(있음))','유전인가(0(없음) or 1(있음))': ")
if  len(datas):
    datas_list=list(map(float,datas.split(',')))
    print(datas_list)

    d=datas.split(',')
    print(d)
    ret=map(float,d)
    print(list(ret))

    # 입력된 정보에 해당하는 탈모 가능성 알려주기
    # 모델의 predict(2D)

    pre_bald=model.predict([datas_list])
    proba=model.predict_proba([datas_list])
    print(f'탈모 가능성은 {(max(proba[0,:])*100)}% {pre_bald[0]} 입니다.') # 지금 pre_iris는 1차원, proba의 경우는 2차원이므로
    #차원 내부로 들어가 줘야 한다.

    model.predict([datas_list])

else:
    print("입력된 정보가 없습니다.")

print()

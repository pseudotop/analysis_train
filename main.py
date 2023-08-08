from lib2to3.pytree import convert
import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

directory = "./dataset"

def list_files_in_directory(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

class FontManager:
    def __init__(self):
        self.path=''
        self.fontprop=''
        self.size=10
        self.titlesize=16

    def set_path(self, _path):
        self.path=_path

    def apply(self):
        self.fontprop = fm.FontProperties(fname=self.path, size=self.size)
        font_name=self.fontprop.get_name()
        plt.rc('font', family=font_name)
        # plt.rc('figure', titlesize=self.titlesize) # figure title 폰트 크기
        plt.rcParams['axes.unicode_minus'] = False #한글 폰트 사용시 마이너스 폰트 깨짐 해결
        plt.rcParams['font.size'] = self.size #기본 폰트 크기
        plt.rcParams['axes.titlesize'] = self.titlesize #제목 폰트 크기

class TrainInfo:
    set_station = {'4.19민주묘지', '가능', '가락시장', '가산디지털단지', '가양', '가오리', '가좌', '가천대', '가평', '간석', '갈매', '강남', '강남구청', '강동', '강동구청', '강매', '강변(동서울터미널)', '강일', '강촌', '개롱', '개봉', '개포동', '개화', '개화산', '거여', '건대입구', '검암', '경기광주', '경마공원', '경복궁(정부서울청사)', '경찰병원', '계양', '고덕', '고려대(종암)', '고색', '고속터미널', '고잔', '곡산', '곤지암', '공덕', '공릉(서울과학기술대)', '공항시장', '공항화물청사', '과천', '관악', '관악산(서울대)', '광나루(장신대)', '광명', '광명사거리', '광운대', '광화문(세종문화회관)', '광흥창(서강)', '교대(법원.검찰청)', '구로', '구로디지털단지', '구룡', '구리', '구반포', '구산', '구성', '구의(광진구청)', '구일', '구파발', '국수', '국회의사당', '군자(능동)', '군포', '굴봉산', '굴포천', '굽은다리(강동구민회관앞)', '금곡', '금릉', '금정', '금천구청', '금촌', '금호', '기흥', '길동', '길음', '김유정', '김포공항', '까치산', '까치울', '낙성대', '낙성대(강감찬)', '남구로', '남동인더스파크', '남부터미널(예술의전당)', '남성', '남영', '남위례', '남춘천', '남태령', '남한산성입구(성남법원.검찰청)', '내방', '노들', '노량진', '노원', '녹번', '녹사평(용산구청)', '녹양', '녹천', '논현', '능곡', '단대오거리', '달월', '답십리', '당고개', '당곡', '당산', '당정', '대곡', '대공원', '대림(구로구청)', '대모산입구', '대방', '대성리', '대야미', '대청', '대치', '대화', '대흥(서강대앞)', '덕계', '덕소', '덕정', '도곡', '도농', '도림천', '도봉', '도봉산', '도심', '도원', '도화', '독립문', '독바위', '독산', '돌곶이', '동대문', '동대문역사문화공원', '동대문역사문화공원(DDP)', '동대입구', '동두천', '동두천중앙', '동묘앞', '동암', '동인천', '동작(현충원)', '두정', '둔촌동', '둔촌오륜', '등촌', '디지털미디어시티', '뚝섬', '뚝섬유원지', '마곡', '마곡나루', '마곡나루(서울식물원)', '마두', '마들', '마석', '마장', '마천', '마포', '마포구청', '망우', '망원', '망월사', '망포', '매교', '매봉', '매탄권선', '먹골', '면목', '명동', '명일', '명학', '모란', '목동', '몽촌토성(평화의문)', '무악재', '문래', '문산', '문정', '미금', '미사', '미아(서울사이버대학)', '미아사거리', '반월', '반포', '발산', '방배', '방이', '방학', '방화', '배방', '백마', '백석', '백양리', '백운', '버티고개', '범계', '별내', '병점', '보라매', '보라매공원', '보라매병원', '보문', '보산', '보정', '복정', '봉명', '봉은사', '봉천', '봉화산(서울의료원)', '부개', '부발', '부천', '부천시청', '부천종합운동장', '부평', '부평구청', '북한산보국문', '북한산우이', '불광', '사가정', '사당', '사릉', '사리', '사평', '산본', '산성', '삼각지', '삼동', '삼산체육관', '삼성(무역센터)', '삼성중앙', '삼송', '삼양', '삼양사거리', '삼전', '상갈', '상계', '상도', '상동', '상록수', '상봉(시외버스터미널)', '상수', '상왕십리', '상월곡(한국과학기술연구원)', '상일동', '상천', '새절(신사)', '샛강', '서강대', '서대문', '서동탄', '서빙고', '서울대벤처타운', '서울대입구(관악구청)', '서울숲', '서울역', '서울지방병무청', '서원', '서정리', '서초', '서현', '석계', '석수', '석촌', '석촌고분', '선릉', '선바위', '선유도', '선정릉', '성균관대', '성수', '성신여대입구(돈암)', '성환', '세류', '세마', '세종대왕릉', '소래포구', '소사', '소요산', '솔밭공원', '솔샘', '송내', '송도', '송정', '송탄', '송파', '송파나루', '수내', '수락산', '수리산', '수색', '수서', '수원', '수원시청', '수유(강북구청)', '수진', '숙대입구(갈월)', '숭실대입구(살피재)', '숭의', '시청', '신갈', '신금호', '신길', '신길온천', '신내', '신논현', '신답', '신당', '신대방', '신대방삼거리', '신도림', '신둔도예촌', '신림', '신목동', '신반포', '신방화', '신사', '신설동', '신용산', '신원', '신이문', '신정(은행정)', '신정네거리', '신중동', '신창(순천향대)', '신촌', '신포', '신풍', '신흥', '쌍문', '쌍용(나사렛대)', '아산', '아신', '아차산(어린이대공원후문)', '아현', '안국', '안산', '안암(고대병원앞)', '안양', '암사', '압구정', '압구정로데오', '애오개', '야당', '야목', '야탑', '약수', '양수', '양원', '양재(서초구청)', '양정', '양주', '양천구청', '양천향교', '양평', '어린이대공원(세종대)', '어천', '언주', '여의나루', '여의도', '여주', '역곡', '역삼', '역촌', '연수', '연신내', '염창', '영등포', '영등포구청', '영등포시장', '영종', '영통', '오금', '오류동', '오리', '오목교(목동운동장앞)', '오목천', '오빈', '오산', '오산대', '오이도', '옥수', '온수(성공회대입구)', '온양온천', '올림픽공원(한국체대)', '왕십리(성동구청)', '외대앞', '용답', '용두(동대문구청)', '용마산', '용마산(용마폭포공원)', '용문', '용산', '우장산', '운길산', '운서', '운정', '운천', '원당', '원덕', '원인재', '원흥', '월계', '월곡(동덕여대)', '월곶', '월드컵경기장(성산)', '월롱', '을지로3가', '을지로4가', '을지로입구', '응봉', '응암', '의왕', '의정부', '이대', '이매', '이수', '이천', '이촌(국립중앙박물관)', '이태원', '인덕원', '인천', '인천공항1터미널', '인천공항2터미널', '인천논현', '인하대', '일산', '일원', '임진강', '잠실(송파구청)', '잠실나루', '잠실새내', '잠원', '장승배기', '장암', '장지', '장한평', '정릉', '정발산', '정부과천청사', '정왕', '정자', '제기동', '제물포', '종각', '종로3가', '종로5가', '종합운동장', '주안', '주엽', '죽전', '중계', '중곡', '중동', '중랑', '중앙', '중앙보훈병원', '중화', '증미', '증산(명지대앞)', '지제', '지축', '지평', '지행', '직산', '진위', '창동', '창신', '천마산', '천안', '천왕', '천호(풍납토성)', '철산', '청구', '청담', '청라국제도시', '청량리(서울시립대입구)', '청명', '청평', '초월', '초지', '총신대입구(이수)', '춘의', '춘천', '충무로', '충정로(경기대입구)', '탄현', '탕정', '태릉입구', '태평', '퇴계원', '파주', '판교', '팔당', '평내호평', '평촌', '평택', '평택지제', '풍산', '하계', '하남검단산', '하남시청(덕풍·신장)', '하남풍산', '학동', '학여울', '한강진', '한남', '한대앞', '한성대입구(삼선교)', '한성백제', '한양대', '한티', '합정', '행당', '행신', '혜화', '호구포', '홍대입구', '홍제', '화계', '화곡', '화랑대(서울여대입구)', '화서', '화전', '화정', '회기', '회룡', '회현(남대문시장)', '효창공원앞', '흑석(중앙대입구)'}
    dict_line = {
            # https://librewiki.net/wiki/%ED%8B%80:%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD_%EC%B2%A0%EB%8F%84_%EB%85%B8%EC%84%A0%EC%83%89
            '1호선': '#0033A0',
            '2호선': '#00B140',
            '3호선': '#FC4C02',
            '4호선': '#30E6FF',
            '5호선': '#A05EB5',
            '6호선': '#C75D28',
            '7호선': '#6D712E',
            '8호선': '#E31C79',
            '9호선': '#ACAA88',

            '우이신설선': '#C7D138',
            '신림선': '#558BCF',
            '동북선': '#B21E36',
            '위례선': '#CCCCCC',
            '위례신사선': '#CCCCCC',
            '서울 경전철': '#787878',

            '인천 1호선': '#759CCE',
            '인천 2호선': '#F5A251',
            
            '경의중앙선': '#72C6A6',
            '경의선': '#72C6A6', #위 경의중앙선
            '중앙선': '#72C6A6', #위 경의중앙선
            '경춘선': '#168C72',
            '수인분당선': '#F2A900',
            '분당선': '#F2A900', #위 수인분당선
            '수인선': '#F2A900', #위 수인분당선
            '경강선': '#0066FF',
            '서해선': '#84BD00',
            '인천국제공항철도': '#33BAFF',
            '공항철도 1호선': '#33BAFF', #위 인천국제공항철도
            '공항철도 직통열차': '#F68A1E',
            '신분당선': '#BA0C2F',
            '신안산선': '#F04938',

            'GTXA': '#AB087D',
            'GTXB': '#234699',
            'GTXC': '#306E5B',

            '의정부경전철': '#FFA100',
            '용인에버라인': '#51E800',
            '김포골드라인': '#AD8605',
            '인천공항 자기부상철도': '#FFAE43',

            '일산선': '#FC4C02', #3호선 연장 구간
            '장한선': '#000000', #새마을,무궁화호 운행
            '과천선': '#30E6FF', #4호선 연장 구간
            '경원선': '#000000', #중앙선,1호선 접속
            '경인선': '#0033A0', #위 1호선
            '경부선': '#0033A0', #위 1호선
            '안산선': '#F2A900', #위 수인분당선
            '9호선2~3단계': '#ACAA88', #위 9호선 연장
        }

    @classmethod
    def validate(cls, _name):
        if cls.dict_line.get(_name) is not None:
            return True
        return False

    @classmethod
    def get_line_color(cls, _name):
        if cls.validate(_name):
            return cls.dict_line[_name]
        else:
            raise ValueError(f'{_name} does not exist')

    @classmethod
    def show_list_station(cls):
        list_station = list(cls.set_station)
        list_station.sort()
        print(list_station)

def pre_processing():
    files=list_files_in_directory(directory=directory)
    df = pd.DataFrame()
    # line_names = set()
    # station_names = set()

    for i,filename in enumerate(files):
        if filename.startswith('.'): continue
        fname=f'{directory}/{filename}'
        with open(fname, 'r', encoding='utf-8') as file:
            # cols : 사용일자,노선명,역명,승차총승객수,하차총승객수,등록일자
            # df = pd.DataFrame(data=file[1:],columns=file[0])
            lines=file.readlines()

            data=[re.sub(r'[\ufeff\n"]', '', line).split(',')[:6] for line in lines]
            # 첫 번째 행(header)
            if i==0: 
                df = pd.DataFrame(data=data[1:], columns=data[0])
            # raw dataset
            else:
                df_extended = pd.DataFrame(data=data[1:], columns=data[0])
                df = pd.concat([df, df_extended])
                # for raw in data[1:]:
                #     line_name = raw[1] #노선명
                #     line_names.add(line_name)
                    # station_name = raw[2] #역명
                    # station_names.add(station_name)
    output = df.pivot_table(index=[data[0][2],data[0][1]],columns=data[0][0],values=data[0][4],aggfunc='sum', fill_value=0)
    return output

def post_processing(_dataframe, _loc):
    output_dir = 'result'
    os.makedirs(output_dir, exist_ok=True)  
    def convert_df_to_period(_opt=None):
        opt = {
            'Raw': 'raw',
            'M': 'monthly',
            'Y': 'yearly'
        }
        df_custom = _dataframe.apply(pd.to_numeric, errors='coerce') # 타입 캐스팅
        df_custom.index = pd.to_datetime(df_custom.index)
        if _opt != 'Raw': 
            df_custom = df_custom.to_period(_opt) # _opt: 'M', 'Y'등
        df_custom = df_custom.groupby('사용일자').agg('sum')
        df_sum = df_custom.sum(axis=1)
        df_custom['합계']=df_sum
        df_custom.to_csv(f'{output_dir}/{_loc}_{opt[_opt]}.csv')

    convert_df_to_period('Raw') #raw
    convert_df_to_period('M')
    convert_df_to_period('Y')
    plt.savefig(f'{output_dir}/{_loc}_하차인원.png',dpi=200)

def draw(_dataset, _line_names, _loc):
    x, y = _dataset # y는 list 형태

    fig, ax = plt.subplots()
    # ax.plot(x, y, 
    #     '-',
    #     color=TrainInfo.get_line_color(_line_names[0]),
    #     label=f'{_line_names[0]}'
    #     )
    ax.stackplot(x, y, 
        colors=[TrainInfo.get_line_color(line_name) for line_name in _line_names],
        labels=_line_names
        )

    ax.set_xlabel(r'Date')
    # Major ticks every half year, minor ticks every month,
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    # Text in the x-axis will be displayed in 'YYYY-mm' format.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax.set_ylabel(r'# of Population')
    # Format y-axis with a comma as thousands separator
    formatter = FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True)

    # ax.set_title(_loc, loc='left', y=0.85, x=0.02)
    ax.set_title(f'하차인원 - {_loc}', y=1.05)
    ax.legend(loc='upper left')
    plt.xticks(rotation=30)

    # plt.show()

def run(_loc_list):
    output = pre_processing()
    loc_list = _loc_list if len(_loc_list) else []
    for _loc in loc_list:
        output_filtered = output.loc[_loc]
        output_filtered = output_filtered.transpose()

        # print(output_filtered)
        line_names = output_filtered.columns.tolist()

        output_filtered.index = pd.to_datetime(output_filtered.index)
        x, y = [output_filtered.index.tolist(),np.transpose(output_filtered.to_numpy(dtype=np.int32).tolist())]
        # print(x[:10], y[:10])
        dataset = (x, y)

        draw(dataset, line_names, _loc)
        post_processing(output_filtered, _loc)
 
font = FontManager()
font.set_path('/Users/bjkim_air/Library/Fonts/NanumSquareR.ttf')
font.apply()

# _loc = '4.19민주묘지'
# _loc = '여의도'
_loc_list = [
    '안국',
    '종로3가',
    '을지로입구',
    '을지로3가',
    '충무로',
    '망원',
    '녹사평(용산구청)',
    '이태원',
    '한강진',
    '노량진',
    '문래',
    '신당',
    '약수',
    '성수',
    '서울숲',
]

run(_loc_list)
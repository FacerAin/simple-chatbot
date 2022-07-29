from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel

p = Preprocess(word2index_dic='train_tools/dict/chatbot_dict.bin')

intent = IntentModel(model_path = 'models/intent/intent-model.pt', preprocess=p)

queries = ["안녕하세요!", "나쁜놈아", "탕수육 하나 주문할게요", "3시에 3명 예약할게요"]

for query in queries:
    predict = intent.predict_class(query)
    predict_label = intent.labels[predict]
    print("Query : ", query)
    print("Prediction Class : ", predict)
    print("Predction Label : ", predict_label)
    print('-------------------------------------------------')
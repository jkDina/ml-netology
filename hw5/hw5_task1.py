#Необходимо для каждой распознанной фамилии 
#вывести на экран наиболее подходящий вариант из стартового протокола
import Levenshtein
claim_list = [
    'Шехавцова Анна',
    'Гречихина Наталья',
    'Козлова Алена',
    'Груздева Алина',
    'Кущенко Анна',
    'Чистякова Анастасия'
]
speech_recognition = [
    'кучменко она',
    'кущенко оксана',
    'груздь алина',
    'рычихина наталья',
    'шиховцева на',
    'чистова анастасия'
]
for recognized_name in speech_recognition:
    distances = {real_name: Levenshtein.distance(recognized_name, real_name) for real_name in claim_list}
    #print(distances)
    print(recognized_name, '-', sorted(distances.items(), key = lambda x: x[1])[0][0])

#hblab
#DB_HOST = 'ocr.cscfmrkvsmgv.ap-southeast-1.rds.amazonaws.com'
#DB_USER = 'ocr'
#DB_PASS = 'av4LFFSQvLgL4ubA'
#DB_NAME = 'ocr_'

#staging
#DB_HOST = '192.168.2.12'
#DB_USER = 'asilla'
#DB_PASS = 'asilla'
#DB_NAME = 'ocr_'

#testerver
DB_HOST = '192.168.2.12:3307'
DB_USER = 'root'
DB_PASS = 'asilla2018'
DB_NAME = 'ocr_'



SAVE_DIR_OCR="static/orc"
SAVE_DIR_IMG="static/img"
SAVE_DIR_TEMP="static/template"

LOG_LEVEL = 10 # ERROR: 40 INFO: 20 DEBUG: 10
VISUAL = True
CROP_PADDING = (5, 5) #PADING (X PIXEL, Y PIXEL)

TEMPLATE_CHAR_TYPE = {
  "1": "numberonly",
  "2": "kanji",
  "3": "address",
  "4": "latin",
  "5": "other",
  "6": "kata",
  "14": "choice",
  "8": "checkbox_rect",
  "9": "checkbox_circle",
  "10": "date",  # Date
  "11": "postcode",  # Post code
  "12": "postcode",  # Telephone
  "13": "currency"   # Currency
}


#Use line number for query template
MIX_RECOGNITION_LINE_NUM=15

#Rate for detection boxs are same line
MIX_LINE_OVERLAP_THRESHOLD=0.5

#setting rate overlap area between template box and detection box
BOX_AREA_OVERLAP_THRESHOLD=0.4

#setting condition for matching template image
MATCHING_TEMPLATE_THRESHOLD=(20, 400, 1200, 0.05)


#serving server information
HOST = "localhost"
HOST2 = "192.168.2.12"
SERVING_SERVER = {
  "detection": [HOST, 8501, "east_detection"],
  "recognize": [HOST, 8502, "recognize_mix_aster"],
  "number": [HOST2, 8503, "number_aster"],
  "date": [HOST, 8503, "date_aster"],
  "numberonly": [HOST, 8504, "number_only_aster"],
  "kata": [HOST, 8505, "kata_aster"],
  "currency": [HOST2, 8504, "currency_aster"],
  "postcode": [HOST2, 8502, "postcode"],
  "latin": [HOST2, 8509, "latin"],
  "choice": [HOST2, 8501, "circle_choice"]
}

#Started  detection.sh  at port  8501
#Started  mix.sh  at port  8502
#Started  circle.sh  at port  8503
#Started  postcode.sh  at port  8504
#Started  kata.sh  at port  8505
#Started  numberonly.sh  at port  8506
#Started  date.sh  at port  8507
#Started  currency.sh  at port  8508

#HOST = "52.183.35.72"
#HOST2 = "52.183.35.72"
#SERVING_SERVER = {
#  "detection": [HOST, 8501, "detection"],
#  "recognize": [HOST, 8502, "mix"],
#  "circle": [HOST2, 8503, "circle"],
#  "postcode": [HOST, 8504, "postcode"],
#  "kata": [HOST, 8505, "kata"],
#  "numberonly": [HOST, 8506, "numberonly"],
#  "date": [HOST2, 8507, "date"],
#  "currency": [HOST2, 8508, "currency"]
#}


#serving server information
#HOST = "52.42.146.192"#Staging
#SERVING_SERVER = {
#  "detection": [HOST, 8501, "east_detection"],
#  "recognize": [HOST, 8502, "recognize_mix_aster"],
#  "number": [HOST, 8503, "number_aster"],
#  "date": [HOST, 8507, "date_aster"],
#  "numberonly": [HOST, 8506, "numberonly_aster"],
#  "currency": [HOST, 8508, "currency_aster"],
#  #"address": [HOST, 8504, "address_aon"],
#  "kata": [HOST, 8505, "kata_aster"],
#  "latin": [HOST, 8502, "recognize_mix_aster"],
#  "postcode": [HOST, 8504, "postcode"],
#  "choice": [HOST, 8503, "circle_choice"]
#}
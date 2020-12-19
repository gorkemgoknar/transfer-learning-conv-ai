import os 
import collections


import pickle
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import pandas as pd
import re

cEXT = pickle.load( open( "data/models/cEXT.p", "rb"))
cNEU = pickle.load( open( "data/models/cNEU.p", "rb"))
cAGR = pickle.load( open( "data/models/cAGR.p", "rb"))
cCON = pickle.load( open( "data/models/cCON.p", "rb"))
cOPN = pickle.load( open( "data/models/cOPN.p", "rb"))
vectorizer_31 = pickle.load( open( "data/models/vectorizer_31.p", "rb"))
vectorizer_30 = pickle.load( open( "data/models/vectorizer_30.p", "rb"))


def is_dialog(text):
  if text.startswith("*****"):
    return False 
  
  if len(text) < 5 :
    ##make sure at least 5 characters
    ##: is guaranteed, plus a \n
    return False 

  if '-------' in text:
    return False
  
  return True

def generate_dialog_holder(lines):
  dialog_holder = []
  dialog = []
  for line in lines:
     if is_dialog(line):
        dialog.append(line)
     elif line.isspace():
        #only whitespace pass it
        continue
     else:
       #make sure at least 2 lines
       if len(dialog)>= 2:
          #only want at least 2 lines
          #single lines may be narrator or title
          dialog_holder.append(dialog)
       
       dialog = []
  
  return dialog_holder 


def generate_dialogs(dialog_holder):
  dialogs = {}
  dialogs["dialogs"] = []
  dialogs["participants"] = set()
  for dialog in dialog_holder:
    chat_dialog = get_chat_dialog(dialog)
    dialogs["dialogs"].append(chat_dialog)
    dialogs["participants"] = dialogs["participants"].union(chat_dialog["participants"])

  return dialogs
    

def preprocess_line(text):
  #add space for ( and ) chars
  ##else everthing sucks
  
  #text = text.replace("("," ( ")
  #text = text.replace(")"," ( ")
  
  #TODO remote text inside ( )
  #these include some action and emotions
  find_between_paranthesis= '\(.*?\)'
  #print( re.findall(find_between_paranthesis,text) ) 
  text = re.sub(find_between_paranthesis ,' ', text)

  #remove multi "-" if exists
  text = text.replace("--"," ")
  
  ##remove empty start end
  ##it is possible text may become empty!
  text = test.strip()

  return text 

  
  
def correct_name(name):
  #check if name has name's voice or name v.o or something
  #just fetch name 
  name = name.lower()
  name = name.replace("'s voice","")
  name = name.replace("v.o.","")
  name = name.replace("v.o","")
  name = name.replace("tv.","")
  name = name.replace("o.s.","")
  name = name.replace("o.s","")
  name = name.replace("(","")
  name = name.replace(")","")
  
  name = name.strip()

  

  #capitalize first word only 
  return name.title()

def get_chat_dialog(dialog):
  ##we have the dialogues
  #now generate utterances.
  chatfile = []
  chat = []

  ##Assume:
  ##2 person chats, name1, name2  if another comes in as name3, assume start conversation again
  #going line by line in dialog

  chat_dialog= {}

  chat = []
  char = collections.defaultdict(list)
  participants = collections.defaultdict(list)

  participant_number = 1
  for line in dialog:
    splitted = line.split(":")
    name = splitted[0].strip()

    name = correct_name(name)
    
    ##max name length 22 (e.g. SOMETHING's CAPTAIN)
    if len(name)>22:
      continue



    talk = " ".join(splitted[1:]).strip()

    talk = preprocess_line(talk)

    if len(talk)> 500:
      #do not get long lines.
      #have token limit of 512
      #find first . after 300th
      startpos=350
      index = talk[startpos:].find('.')
      #if it finds position will be startpos+index, else it will be startpos-1
      print(f"LONG LINE split after pos {startpos+index}: " + talk)
      
      talk = talk[startpos+index: ]

    if len(talk)<2:
      ##there is nothing here
      ##bypass this line
      continue
    



    ##count number of tokens.. Currently it is adjusted for 512
    ##longer than 512 split it 

     
    ##TODO check name starts with in participants
    ##if name (VO) name (..) or name/Something than assume same
    ##as voice over are here too.

    if name not in participants:
      participants[name] = participant_number
      participant_number += 1 
          

    chat.append([name, participants[name],  talk])
    char[name].append(talk)

  chat_dialog["chat"] = chat
  chat_dialog["char"] = char
  chat_dialog["participants"] = participants

  return chat_dialog



def generate_dialog_from_file(filename):
  with open(filename) as f:
    lines = f.readlines()
    dialog_holder = generate_dialog_holder(lines)
    

  dialogs = generate_dialogs(dialog_holder)
  dialogs["filename"] = os.path.basename(filename)

  print(f"File: { dialogs['filename'] } , Participant count: { len(dialogs['participants']) } " )
  return dialogs

 
def predict_personality(text):
    scentences = re.split("(?<=[.!?]) +", text)
    text_vector_31 = vectorizer_31.transform(scentences)
    text_vector_30 = vectorizer_30.transform(scentences)
    EXT = cEXT.predict(text_vector_31)
    NEU = cNEU.predict(text_vector_30)
    AGR = cAGR.predict(text_vector_31)
    CON = cCON.predict(text_vector_31)
    OPN = cOPN.predict(text_vector_31)
    return [EXT[0], NEU[0], AGR[0], CON[0], OPN[0]]


def get_all_text_for_char(name,dialogs):
  #name = 'HAL'
  char_text = []
  for d in dialogs['dialogs'] :
    text_arr = d['char'][name]
    if len(text_arr) >0:
      char_text.append( d['char'][name] )

  all_char_text = " ".join( [" ".join(t) for t in char_text] )
  return all_char_text



def generate_dialog_and_personality(file, debug=False):
  dialogs = generate_dialog_from_file(file)


  dialogs["personalities"] = collections.defaultdict(list)
  for name in dialogs["participants"]:
    
    all_char_text = get_all_text_for_char(name, dialogs)
    personality = predict_personality(all_char_text)
    if debug: print(f"{name} : Personality: {personality} ")

    dialogs["personalities"][name] = personality

    #df = pd.DataFrame(dict(r=predictions, theta=['EXT','NEU','AGR', 'CON', 'OPN']))
    #fig = px.line_polar(df, r='r', theta='theta', line_close=True) 
    #fig.show()

  return dialogs





scripts = collections.defaultdict(list)

path = '/content/chatbot/movie_scripts/scriptcleaned/'
for file in os.listdir(path):
  dialogs = generate_dialog_and_personality(path+'/'+file)
  scripts["movies"].append(dialogs)




file = '/content/chatbot/movie_scripts/scriptcleaned/cleaned_2001.txt'

dialogs = generate_dialog_and_personality(file)



#number of chats containing only 2 participants
two_person_chats = []
for id, chat in enumerate( dialogs["dialogs"] ) :
  num_chatter= len ( chat['participants'] ) 
  #print(num_chatter)
  if num_chatter==2:
    two_person_chats.append([id,chat])




import random 
def get_random_line_said_by_char(name,dialogs,current_recursion=0):
  max_dialogs = len( dialogs["dialogs"]) 

  #names are title mode
  name = name.title()

  dialog_with_name = None
  ##loop on dialogs
  for d in dialogs["dialogs"]:
    
    if name not in d['participants']:
      #name not here 
      continue
    else:
      dialog_with_name = d
      break

  if dialog_with_name is None:
    #should not happen
    print("Probably you gave incorrect name")
    return "I am no one."
    

  chat =  dialog_with_name["chat"]
  sample = random.sample(chat, 1) 

  if sample[0][0] != name :
    ##recursive! as my hit same char
    current_recursion += 1 
    if current_recursion > 10:
      ##enough already 
      return "I could not find anything to say."
    else:
      ##2th has line
      return get_random_line_said_by_char(name,dialogs,current_recursion) 
  
  #print(f"{current_recursion} recursions")
  #print(sample)
  return sample[0][2]


def get_random_line_not_said_by_char(name,dialogs,current_recursion=0):
  max_dialogs = len( dialogs["dialogs"]) 

  random_dialog_id = random.randint(0,max_dialogs-1)
  #pick one of the chat character makes
  chat =  dialogs["dialogs"][random_dialog_id]["chat"]
  sample = random.sample(chat, 1) 

  if sample[0][0] == name :
    ##recursive! as my hit same char
    current_recursion += 1 
    if current_recursion > 5:
      ##enough already 
      return "I could not find anything to say."
    else:
      ##2th has line
      return get_random_line_not_said_by_char(name,dialogs,current_recursion) 
  
  #print(f"{current_recursion} recursions")
  #print(sample)
  return sample[0][2]




##build history from chat
##Name, speaker_id_in_file, line
#two_person_chats[5][1]['chat']


def get_utterance_list(chat, dialogs, char_opening_lines=None, num_candidates=10, use_personality_score=False):
  
  if char_opening_lines is None:
    char_opening_lines = collections.defaultdict(list)
  
  chat_length = len(chat) 

  ##build utterance list 

  utterances = []

  for id, line in enumerate( chat ):
    ##check if next line exists:
    if id < (chat_length-1):
      next_line = chat[id+1]
    else:
      break

    name = line[0]

    if id == 0:
      #  ##first line no history
      #  ##add this to opening line for this char
      char_opening_lines[name].append(line[2])


    #print("\n")
    #print("History:\n" + str(chat[0:id]) )
    #print("Line:\n" + str(line) )
    #print("Next line:\n" + str(next_line))
    
    #TODO here " " spaces must be deleted for each line
    ##and ignore those who have no lines!!
    ##some text can not be properly parsed  
    history = [line[2] for line in chat[0:(id+1)]]

    name_under_test = next_line[0]
    real_response = next_line[2]
    
    candidates = []
    for _ in range(num_candidates):
        ##WARNING HARDOCODEC USE OF DIALOGS!! IT must be accessible here
        candidates.append( get_random_line_not_said_by_char(name_under_test,dialogs) ) 

    candidates.append(real_response)

    current_utterance = {}
    current_utterance["candidates"] = candidates
    current_utterance["history"] = history
    current_utterance["name"] = name_under_test
    
    utterances.append(current_utterance)


  #for each utterance generate personality 
  for utterance in utterances:
    name = utterance["name"]

    personality = []

    if use_personality_score:
      personality_traits= dialogs["personalities"][name]

      p_openness  = "inventive curious" if  personality_traits[0] == 1 else "consistent cautious"
      p_conscientiousness = "efficient organized" if personality_traits[1] == 1 else "easy-going careless"
      p_extraversion = "outgoing energetic" if personality_traits[2] == 1 else "solitary reserved"
      p_agreebleness = "friendly compassionate" if personality_traits[3] == 1 else "challenging detached"
      p_neuroticism = "sensitive nervous" if personality_traits[4] == 1 else "secure confident"

      personality.append("I am " + p_openness)
      personality.append("I am " +p_conscientiousness)
      personality.append("I am " +p_extraversion)
      personality.append("I am " +p_agreebleness)
      personality.append("I am " +p_neuroticism)
    else:
      char_line = get_random_line_said_by_char(name,dialogs)
      if len(char_line)> 100:
        startpos=90
        index = l[startpos:].find('.')
        char_line = char_line[startpos+index: ]
      personality.append(char_line)
    
    personality.append("My name is " + name +".")
    utterance["personality"] = personality
                    
  return utterances, char_opening_lines




  #char_opening_lines = collections.defaultdict(list)
#chat= two_person_chats[10][1]['chat']

all_utterances = [] 
char_opening_lines = collections.defaultdict(list)
for chats in two_person_chats:
  chat = chats[1]['chat']
  utterances, char_opening_lines= get_utterance_list(chat,dialogs, char_opening_lines=char_opening_lines,  num_cadidates=10)
  all_utterances.append(utterances)


from collections import defaultdict
persona = {}
for ut_dialog in all_utterances:
  ##loop ut
  for ut in ut_dialog:
    name = ut["name"]
    if name not in persona:
      persona[name] = {}

    persona[name]["personality"] = ut["personality"]
    if 'utterances' not in persona[name]:
      persona[name]["utterances"] = []

    persona[name]["utterances"].append(ut)
    







def get_persona_from_file(file):
 
  dialogs = generate_dialog_and_personality(file)


  #number of chats containing only 2 participants
  two_person_chats = []
  for id, chat in enumerate( dialogs["dialogs"] ) :
    num_chatter= len ( chat['participants'] ) 
    #print(num_chatter)
    if num_chatter==2:
      two_person_chats.append([id,chat])

  print("Two person chat count:{}".format(len(two_person_chats)))
  
  all_utterances = [] 
  char_opening_lines = collections.defaultdict(list)
  for chats in two_person_chats:
    chat = chats[1]['chat']
    utterances, char_opening_lines= get_utterance_list(chat,dialogs, char_opening_lines=char_opening_lines,  num_cadidates=10)
    all_utterances.append(utterances)


  persona = {}
  for ut_dialog in all_utterances:
    ##loop ut
    for ut in ut_dialog:
      name = ut["name"]
      if name not in persona:
        persona[name] = {}

      persona[name]["personality"] = ut["personality"]
      if 'utterances' not in persona[name]:
        persona[name]["utterances"] = []

      persona[name]["utterances"].append(ut)
    
  return list(persona.values())



from collections import defaultdict
personachat = {}

##file = '/content/chatbot/movie_scripts/scriptcleaned/cleaned_2001.txt'

path = '/content/chatbot/movie_scripts/scriptcleaned'
for file in os.listdir(path):
  basefile = os.path.basename(file)
  filepath = path + "/" + file
  print(file)
  personachat[basefile] = get_persona_from_file(filepath)

#personachat["2001"] = list( persona.values() ) 

persona =[]
for name,val in personachat.items():
  for p in val:
    persona.append(p)



personachat = {}
train_split_index = round(len(persona)*0.8)
personachat["train"] = persona[0:train_split_index]
personachat["val"] = persona[train_split_index:]




##SAVE TO JSON
import json
#a_dict = {'new_key': 'new_value'}
#with open('test.json') as f:
#    data = json.load(f)
#data.update(a_dict)
all_utterances = {"persona" : all_utterances}
persona_file = '/content/chatbot/moviepersonafile.json'
with open(persona_file, 'w') as f:
    json.dump(all_utterances, f)


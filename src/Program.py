import mysql.connector
from collections import Counter
import spacy
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from wordcloud import WordCloud
from PIL import Image as img
from empath import Empath
from nltk.tokenize import word_tokenize
import gensim
from pprint import pprint  
import gensim.corpora as corpora
import string
from tkinter import *
def plot_most_negative_words(data,title):
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('words');
    plt.hist(data, bins=np.arange(6)-0.5, edgecolor='black')
    plt.show()   
def negative_common_words(negative_barrier):
    most_frequent = []
    tokenized_barrier = []
    for sentence in negative_barrier:
        tokenized_barrier=tokenized_barrier + preprocess_sentence(sentence)
    final_pollution = [''.join(c for c in s if c not in string.punctuation) for s in tokenized_barrier]
    final_pollution = [s for s in final_pollution if s]
    for i in range(1,15):
          most_common_words= [word for word, word_count in Counter(final_pollution).most_common(i)]
          for word in final_pollution:
              if word == most_common_words[i-1] and (word != 'I' and word != 'The' and word != 'A' and word != 'ba' and word != 'In' and word != 'It' and word != 'And' and word != 'If'):
                  most_frequent.append(word)
    return most_frequent
def plot_trade_barrier(data,title):
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('barrier');
    plt.hist(data, bins=np.arange(6)-0.5, edgecolor='black')
    plt.show()
def trade_barrier_result(pollution_dictionary,cost_dictionary,technology_dictionary,transportation_dictionary,competition_dictionary,answers):
    result_array = []
    pollution_sentences = []
    cost_sentences = []
    technology_sentences = []
    transportation_sentences = []
    competition_sentences = []
    for sentence in answers:
        filtered_sentence = preprocess_sentence(sentence)
        flag_pol=False
        flag_cost=False
        flag_tec=False
        flag_tran=False
        flag_comp=False
        for x in filtered_sentence:
                for j in pollution_dictionary:
                    Ratio = levenshtein_ratio_and_distance(x.lower(),j.lower(),ratio_calc = True)
                    if Ratio >= 0.85:
                        flag_pol=True
        for x in filtered_sentence:
                for j in cost_dictionary:
                    Ratio = levenshtein_ratio_and_distance(x.lower(),j.lower(),ratio_calc = True)
                    if Ratio > 0.85:
                        flag_cost=True
        for x in filtered_sentence:
                for j in technology_dictionary:
                    Ratio = levenshtein_ratio_and_distance(x.lower(),j.lower(),ratio_calc = True)
                    if Ratio > 0.85:
                        flag_tec=True
        for x in filtered_sentence:
                for j in transportation_dictionary:
                    Ratio = levenshtein_ratio_and_distance(x.lower(),j.lower(),ratio_calc = True)
                    if Ratio > 0.85:
                        flag_tran=True    
        for x in filtered_sentence:
                for j in competition_dictionary:
                    Ratio = levenshtein_ratio_and_distance(x.lower(),j.lower(),ratio_calc = True)
                    if Ratio > 0.85:
                        flag_comp=True
        if flag_pol==True:
            result_array.append("pollution")
            pollution_sentences.append(sentence)
        if flag_cost==True:
            result_array.append("cost")
            cost_sentences.append(sentence)
        if flag_tec==True:
            result_array.append("technology")
            technology_sentences.append(sentence)
        if flag_tran==True:
            result_array.append("transportation")
            transportation_sentences.append(sentence)
        if flag_comp==True:
            result_array.append("competition")
            competition_sentences.append(sentence)
    return result_array,pollution_sentences,cost_sentences,technology_sentences,transportation_sentences,competition_sentences
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
def lda_analysis(data):
    filtered_sentence=[]
    for sentence in answers:
            filtered_sentence.append(preprocess_sentence(sentence))      
    data_words = list(sent_to_words(filtered_sentence))
    
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    
    # Create Corpus
    texts = data_words
    
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # View
    #print(corpus[:1])
    
    # Human readable format of corpus (term-frequency)
    #print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
    
    #We have everything required to train the LDA model. In addition to the corpus and dictionary, you need to provide the number of topics as well.
    #Apart from that, alpha and eta are hyperparameters that affect sparsity of the topics. According to the Gensim docs, both defaults to 1.0/num_topics prior.
    #chunksize is the number of documents to be used in each training chunk. update_every determines how often the model parameters should be updated and passes is the total number of training passes.
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=5, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])
def get_answers(h,usr,pwd):
    mydb = mysql.connector.connect(
      host=h,
      user=usr,
      password=pwd,
      database="t_barrier"
    )    
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT Text FROM answers")
    myresult = mycursor.fetchall()
    return myresult
def preprocess_sentence(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence[0]) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return filtered_sentence
def imp_exp_result(import_dictionary,export_dictionary,answers):
    result_array = []
    imp_exp_sentences = []
    for sentence in answers:
        filtered_sentence = preprocess_sentence(sentence)
        flag_i=False
        flag_e=False
        for x in filtered_sentence:
                for j in import_dictionary:
                    Ratio = levenshtein_ratio_and_distance(x.lower(),j.lower(),ratio_calc = True)
                    if Ratio >= 0.85:
                        flag_i=True
        if flag_i!=True:
            for x in filtered_sentence:
                    for j in export_dictionary:
                        Ratio = levenshtein_ratio_and_distance(x.lower(),j.lower(),ratio_calc = True)
                        if Ratio > 0.85:
                            flag_e=True
        if flag_i==True:
            result_array.append("import")
            imp_exp_sentences.append(sentence)
        if flag_e==True:
            result_array.append("export")
            imp_exp_sentences.append(sentence)
    return result_array,imp_exp_sentences
def named_entity_result(answers):
    named_entities_sentences = []
    nlp = spacy.load("en_core_web_sm")
    result_Array=[]
    most_frequent=[]
    for sentence in answers:
         flag_one_entity=False
         doc = nlp(sentence[0])
         for ent in doc.ents:
             if ent.label_ == 'GPE':
                 if ent.text!='Tolyasha' and ent.text!='Zinna' and ent.text!='Rocket' and ent.text!='Zinna' and ent.text!='Zikawa' and ent.text!='Zabihli':## nickname filter
                     result_Array.append(ent.text)
                     if flag_one_entity==False:
                         named_entities_sentences.append(sentence)
                         flag_one_entity=True
    for i in range(1,31):
         most_common_words= [word for word, word_count in Counter(result_Array).most_common(i)]
         for location in result_Array:
             if location == most_common_words[i-1]:
                 most_frequent.append(location)
    return most_frequent,named_entities_sentences 
def plot_imp_exp(data):
    plt.title('Exports threads and import threads')
    plt.ylabel('Number of threads')
    plt.xlabel('Kind of activity');
    plt.hist(data, bins = 3, edgecolor='black')
    plt.show()
def plot_named_entities(data):
    plt.title('Foreign entities')
    plt.ylabel('Frequency')
    plt.xlabel('Locations');
    plt.xticks(rotation=90)
    plt.hist(data, bins=np.arange(31)-0.5, edgecolor='black')
    plt.show()
def sentiment_result(data):
    result_array = []
    negative = []
    for sentence in data:
        sid_obj = SentimentIntensityAnalyzer() 
        sentiment_dict = sid_obj.polarity_scores(sentence) 
        if sentiment_dict['compound'] >= 0.05 : 
            result_array.append("positive")
        elif sentiment_dict['compound'] <= - 0.05 : 
            result_array.append("negative")
            negative.append(sentence)
        else : 
            result_array.append("neutral")
    return result_array,negative
def plot_sent_analysis(data,title):
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('Sentiment');
    plt.hist(data, bins=np.arange(4)-0.5, edgecolor='black')
    plt.show()
# def plot_cloud(wordcloud):
#     # Set figure size
#     plt.figure(figsize=(40, 30))
#     # Display image
#     plt.imshow(wordcloud) 
#     # No axis details
#     plt.axis("off");          
def wordcloud(data):
    word_cloud_graph=[]
    for i in range(1,len(data)):
        new_element=list(data[i])
        word_cloud_graph.append(''.join(new_element))
    cloud=''.join(word_cloud_graph)
    # Import image to np.array
    mask = np.array(img.open('mask.png'))
    # Generate wordcloud
    wordcloud = WordCloud(width = 6000, height = 4000, random_state=4, background_color='white', colormap='Blues', collocations=False, stopwords = list(stopwords.words('english'))+["Message","will","good","one","little","ba","know","want"]+get_date_entities(answers), mask=mask).generate(cloud)
    # Plot
    #plot_cloud(wordcloud)
    wordcloud.to_file("wordcloud.png")
def get_date_entities(data):
    nlp = spacy.load("en_core_web_sm")
    result_Array=[]
    for sentence in answers:
         doc = nlp(sentence[0])
         for ent in doc.ents:
             if ent.label_ == 'DATE':
                     result_Array.append(ent.text)
    return result_Array
def empath_analysis(data):
    word_cloud_graph=[]
    for i in range(1,len(answers)):
       new_element=list(answers[i])
       word_cloud_graph.append(''.join(new_element))
    cloud=''.join(word_cloud_graph)
    lexicon = Empath()
    prova=lexicon.analyze(cloud, normalize=True)
    values=[]
    for x in prova:
        values.append(prova[x])
    categories=sorted(prova.items(), key=lambda x: x[1], reverse=True)
    #12 categories
    for i in range(1,13):
        print(categories[i])
#----------------------------------------------------------------
import_dictionary = ["import","imported","importation","importee"]
export_dictionary = ["export","exported","exportation","exportee"]
pollution_dictionary = ["Pollution","Oxygen","Waste","Biomass","Clean","Temperature","Climate","Air","Recycle","Ecology"]
cost_dictionary = ["Cost","Expensive","Money","Profit","Price","Salary","€","Sale", "Buy","Loss","Pay","Economic","Investment","Discount","Budget"]
technology_dictionary = ["Technology","Ultrasound","Pump","Automation","Mechanisation","High-Technology","Computer","Robot","Laboratory","Development","Technique","Hydrobiology","Incubator","Tool","Treatment","Power","Equipment","Filter","Industry","Engine","Acquaculture","Material","Heat","Transgenic"]
transportation_dictionary = ["Transportation","Road","Village","Canals","Channels","Car","Transport","Delivery","Container","Drive","Region","Border","City"]
competition_dictionary = ["Competition","Business","Countries","Leaders","Shop","Commerce","Customer","Trade","Domestic","Advertise","Network","Market"]

print("Before proceeding be sure to have access to the database t_barrier, proceed now to insert the needed informations to access the database")
host = input("Insert the location where the database is hosted(eg. 127.0.0.1):")
user = input("\nInsert the user(eg. root):")
pwd = input("\nInsert the password:")
connected=1;
try:
    answers = get_answers(host,user,pwd)
except:
    print("There was a problem within the connection")
    connected=0
# creating the tkinter window 
Main_window = Tk() 
  

# function define for  
# updating the my_label 
 
def task1(): 
    
    # use global variabel 
    my_text = "Task1 has been completed.You will be able to see the plot on the *plot* section of the IDE .\n Task 1 description: We would like to find out whether export / import is relevant in the discussion content. \nFor this purpose, sketch a set of semantically equivalent & inflection words to export / import using WordNet synonymy and/other vocabulary of your choice. \nPerform a simple string matching and to draw the histogram that shown the number of threads associated to export and number of threads associated to import."
      
    # configure 
    my_label.config(text = my_text) 
    
    plot_imp_exp(imp_exp_result(import_dictionary,export_dictionary,answers)[0])
  
def task2(): 
    
    # use global variabel 
    my_text = "Task2 has been completed.You will be able to see the plot on the *plot* section of the IDE .\n Task 2 description: Further to string matching, use SpaCy named-entity tagger to identify the location based named-entities and plot histogram indicating the number of threads that mention the underlined named-entity (You may restrict to the thirty most frequent location entities).\n Manually identify the foreign named-entities (outside Russia) and comment on the importance of the foreign trade in the discussion."
    plot_named_entities(named_entity_result(answers)[0])
    # configure 
    my_label.config(text = my_text) 

def task3(): 
    
    # use global variabel 
    my_text = "Task3 has been completed.You will be able to see the plot on the *plot* section of the IDE .\n Task 3 description: We would like to find out how the discussion approached foreign trade either positively or negatively. \nFor this purpose, concatenate all threads identified either through import / export string matching or foreign location named-entity recognition and perform sentiment los using SentiStrength or another sentiment software of your choice.\n Draw the histogram showing the number of threads associated to positive sentiment and that of negative sentiment."
    plot_sent_analysis(sentiment_result(imp_exp_result(import_dictionary,export_dictionary,answers)[1]),"Import export sentiment analysis")
    plot_sent_analysis(sentiment_result(named_entity_result(answers)[1]),"Foreign location sentiment analysis")
    # configure 
    my_label.config(text = my_text) 
    
def task4(): 
    
    # use global variabel 
    my_text = "Task4 has been completed.You will be able to see the wordmap at the address FinnishRussianTradeBarrier\src\wordcloud.png.\n Task 4 description: After appropriate processing stage where you remove all data associated to date, headline and stopwords, plot the WordCloud graph to visualize the important wording in the discussion forums.\n Comment on the inherent content of the website."
    wordcloud(answers)
    # configure 
    my_label.config(text = my_text) 
    
def task5(): 
    
    # use global variabel 
    my_text = "Task5 has been completed.You will be able to see the categories on the *Editor* section of the IDE.\nTask 5 description: Perform Empath Client https://github.com/Ejhfast/empath-client to the above filtered document and generate the main categories present in the dataset."
    empath_analysis(answers)
    # configure 
    my_label.config(text = my_text) 
    
def task6(): 
    
    # use global variabel 
    my_text = "Task6 has been completed.You will be able to see the topics on the *Console* section of the IDE.\nTask 6 description: Perform LDA with 5 topics and 10 words per topic.\n Manually comment on the nature and type of topic generated by manually trying to find a common meta-topic that link the generated ten-words.  "
    lda_analysis(answers)  
    # configure 
    my_label.config(text = my_text) 
    
def task7(): 
    
    # use global variabel 
    my_text = "Task7 has been completed. You will be able to see the plot on the *Plots* section of the IDE.\n Task 7 description: We want to use categories which are more suitable to trade-barriers. The latter are classified according to Word Economic Organization, which classifies barriers as either natural or artificial (created by state to avoid competition).\n We restrict to natural class barriers. We create subclasses of natural barriers corresponding to Technology, Cost, Pollution, Transportation and Competition. \nFor each of these categories, sketch a set of keywords that best describe the content of each category. You may take a quick look to the content of the database to see the wording employed by the users.\n Perform standard string matching to retrieve threats associated to each of the five categories. Plot a histogram showing the number of threads per category."
    plot_trade_barrier(trade_barrier_result(pollution_dictionary,cost_dictionary,technology_dictionary,transportation_dictionary,competition_dictionary,answers)[0], "Natural trade barriers")  
    # configure 
    my_label.config(text = my_text) 
    
def task8(): 
    
    # use global variabel 
    my_text = "Task8 has been completed. You will be able to see the plot on the *Plots* section of the IDE.\n Task8 description: For each thread in task7, perform sentiment analysis to identify whether it is positive or negative sentiment.\n Store the result in a database."
    result=trade_barrier_result(pollution_dictionary,cost_dictionary,technology_dictionary,transportation_dictionary,competition_dictionary,answers)
    plot_sent_analysis(sentiment_result(result[1])[0],"Pollution trade barrier")
    plot_sent_analysis(sentiment_result(result[2])[0],"Cost trade barrier")
    plot_sent_analysis(sentiment_result(result[3])[0],"Technology trade barrier")
    plot_sent_analysis(sentiment_result(result[4])[0],"Transportation trade barrier")
    plot_sent_analysis(sentiment_result(result[5])[0],"Competition trade barrier")
    # configure 
    my_label.config(text = my_text) 
    
def task9(): 
    
    # use global variabel 
    my_text = "Task9 has been completed.You will be able to see the plot on the *Plots* section of the IDE.\n Task9 description: We want to comprehend the negative sentiment occurrence.\n For this purpose, for each category, gather all threads associated to negative sentiment in a single document and draw the histogram of the most frequent terms, excluding stopword and noisy terms (i.e., uncommon characters, words of less than 2 characters length, etc..)."
    result=trade_barrier_result(pollution_dictionary,cost_dictionary,technology_dictionary,transportation_dictionary,competition_dictionary,answers)

    negative_pollution = sentiment_result(result[1])[1]
    negative_cost = sentiment_result(result[2])[1]
    negative_technology = sentiment_result(result[3])[1]
    negative_transportation = sentiment_result(result[4])[1]
    negative_competition = sentiment_result(result[5])[1]

    plot_most_negative_words(negative_common_words(negative_pollution),"negative words in pollution topics")
    plot_most_negative_words(negative_common_words(negative_cost),"negative words in cost topics")
    plot_most_negative_words(negative_common_words(negative_technology),"negative words in technology topics")
    plot_most_negative_words(negative_common_words(negative_transportation),"negative words in transportation topics")
    plot_most_negative_words(negative_common_words(negative_competition),"negative words in competition topics")  
    # configure 
    my_label.config(text = my_text) 

# create a button widget and attached    
# with counter function    
my_button1 = Button(Main_window, 
                   text = "TASK1", 
                   command = task1) 
my_button2 = Button(Main_window, 
                   text = "TASK2", 
                   command = task2)
my_button3 = Button(Main_window, 
                   text = "TASK3", 
                   command = task3)
my_button4 = Button(Main_window, 
                   text = "TASK4", 
                   command = task4)
my_button5 = Button(Main_window, 
                   text = "TASK5", 
                   command = task5)
my_button6 = Button(Main_window, 
                   text = "TASK6", 
                   command = task6)
my_button7 = Button(Main_window, 
                   text = "TASK7", 
                   command = task7)
my_button8 = Button(Main_window, 
                   text = "TASK8", 
                   command = task8)
my_button9 = Button(Main_window, 
                   text = "TASK9", 
                   command = task9)
# create a Label widget 
my_label = Label(Main_window, 
                 text = "Press one botton to start the relative task. A single task can take from few seconds to few minutes.") 
  
# place the widgets  
# in the gui window 
my_label.pack() 
my_button1.pack() 
my_button2.pack()
my_button3.pack()
my_button4.pack()
my_button5.pack()
my_button6.pack()
my_button7.pack()
my_button8.pack()
my_button9.pack()
# Start the GUI  
if connected==1:
    print("\nYou are now connected to the database and ready to use our application with the GUI")
    Main_window.mainloop()
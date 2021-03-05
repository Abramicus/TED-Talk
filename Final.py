#!/usr/bin/env python
# coding: utf-8

# # Importing our dataset and Preparing the Data
# 
# *Data set course: https://www.kaggle.com/rounakbanik/ted-talks*
# 
# *Fillm_date and published_date collumns are in Unix timpestamp date format, we need to change them into human readeable one. We will use datetime package* 

# In[35]:


import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import linregress
import numpy

#loading up the csv file
ted = pd.read_csv("D:/PYTHON PROJECT/ted_main.csv")

#checking the columns
print(ted.columns)

#checking the row count
print('Number of rows in the data set: '+str(len(ted)))

#reordering our columns depending on our preferences 
ted = ted[['main_speaker','title','event','comments',
           'views','duration', 'languages', 'num_speaker', 
           'film_date','published_date','speaker_occupation',  
           'tags','ratings','name','url','description', 'related_talks' ]]

#fixing the two timestamp columns
ted['film_date'] = ted['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
ted['published_date'] = ted['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))

#seeing the final table
ted.head()


# # Overall analysis about TED Talks
# 
# ## Number of filmed talks over the years

# In[36]:


ted['year'] = ted['film_date'].apply(lambda x: x.split('-')[2])
year_ted = pd.DataFrame(ted['year'].value_counts().reset_index())
year_ted.columns = ['year', 'talks']
year_ted = year_ted.sort_values('year', ascending=True)
plt.figure(figsize=(18,5))
sb.pointplot(x='year', y='talks', data=year_ted)


# ## Audience participation: views and comments

# In[37]:


print('View statistics:')
print(round(ted['views'].describe()))
print('--------------------------------------')
print('Comment statistics:')
print(round(ted['comments'].describe()))


fig, axs = plt.subplots(ncols=2, figsize=(15,5))
sb.histplot(ted[ted['views'] < 4000000] ['views'], ax=axs[0])
axs[0].axvline(ted['views'].mean(), color='r', linestyle='--')
sb.histplot(ted[ted['comments'] < 1000]['comments'], ax=axs[1])
axs[1].axvline(ted['comments'].mean(), color='r', linestyle='--')
plt.legend({'Mean':ted['comments'].mean()})


# # Interesting insights from the TED data
# 
# ## Who are the best Speakers and what are their occupation?

# In[38]:


#We will sepect the necessary columns and find the best speakers in terms of views
tedvso = ted[['main_speaker', 'views', 'speaker_occupation']].sort_values('views', ascending = False)[:10]
print(tedvso)
tedvsoplt = tedvso.plot.bar(x='main_speaker', y='views', xlabel='Speakers', 
                            ylabel='Views in millions', rot=0, figsize=(20,5), 
                            title='Most Popular Speakers Based on Views', grid=False, legend=False)


# In[39]:


#We will find the occupations of the speakers
get_ipython().run_line_magic('matplotlib', 'inline')
views = tedvso['views']
occupation = numpy.array(tedvso['speaker_occupation'])

plt.rcParams.update({'font.size': 14})

#We are making our pie chart equally distributed
plt.axis("equal")
plt.pie(views, labels=occupation,radius=3, shadow =True, explode=[0.2,0,0,0,0,0,0,0,0,0.4], autopct = '%.2f%%')
fig.set_size_inches(1, 1)

#In order to remove extra lables
plt.show


# ## Which occupation should you choose if you want to become a TED Speaker?

# In[40]:


occupation_ted = ted.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'comments']]
occupation_ted.columns = ['occupation', 'appearances']
occupation_ted = occupation_ted.sort_values('appearances', ascending=False)

plt.rcParams.update({'font.size': 12.5})
plt.figure(figsize=(15,5))
sb.barplot(x='occupation', y='appearances', data=occupation_ted.head(10))
plt.show()


# ## What is the Best duration for a TED talk?

# In[41]:


tedvd = ted[['duration', 'views']].sort_values('duration', ascending = False)
print(tedvd)
tedvdplt = tedvd.plot(x='duration', y='views', xlabel='Time in secs', 
                      ylabel='Views in millions', rot=0, figsize=(20,5), 
                      title='Durations Based on Views', grid=False, legend=False)


# In[42]:


#Best times are between 0 and 1500 (or up to 25mins) , so shorten the time range to 2000 secs
tedvd1 = tedvd[tedvd['duration']<=2000]
print(tedvd1)
tedvdplt1 = tedvd1.plot(x='duration', y='views', xlabel='Time in secs', 
                        ylabel='Views in millions', rot=0, figsize=(20,5), 
                        title='Best Durations Based on Views', grid=False, legend=False)


# ### *Conclustion: The best duration of a TED Talk speech seems to be from 15 to 25 mins.*

# ## Do language options matter, do they impact video views?

# In[43]:


#Top and Least viewed videos and their language options
tedlv_top = ted[['languages', 'views']].sort_values('views', ascending = False)[:15]
print('Top viewed videos and their translations')
print(tedlv_top)
print()
tedlv_low = ted[['languages', 'views']].sort_values('views', ascending = True)[:15]
print('Least viewed videos and their translations')
print(tedlv_low)


# In[44]:


#Language and Views
tedlv = ted[['languages', 'views']]

#Making a correlation matrix and p-value analysis for lnguage and views
cor_mat=tedlv.corr()
print(cor_mat)
print('-----------------------------------------------')
pval=linregress(tedlv['languages'], tedlv['views'])
print(pval)

#Making a scatter plot with a regression line
sb.set(rc={'figure.figsize':(8,8)})
sb.regplot(x="languages", y="views", ci=None, data=tedlv, scatter_kws={"color": "blue"}, 
           line_kws={"color": "red"}).set(title="Does the number of Translations impact View count?")


# ### *Conclusion: No, language options don't seem to matter. They have a very weak relationship with Views.*

# ## What are the most popular TED talk themes?

# In[45]:


#formatting 'tags' for easiler analysis
tedtv = ted[['tags', 'views']].sort_values('views', ascending = False)
tedtv['tags'] = tedtv['tags'].str.split(',')
tedtv


# In[46]:


#splitting tags into separate rows, and removing symbols
tedtv1 = (tedtv.set_index(['views'])['tags'].apply(pd.Series).stack().reset_index().drop('level_1', axis=1).rename(columns={0:'themes'}))
tedtv1['themes'] = tedtv1['themes'].replace({"'":""}, regex=True)
tedtv1['themes'] = tedtv1['themes'].replace({"\[":""}, regex=True)
tedtv1['themes'] = tedtv1['themes'].replace({"\]":""}, regex=True)
tedtv1['themes'] = tedtv1['themes'].replace({" ":""}, regex=True)
print(tedtv1)

print('-----------------------------------------------')
print('Number of unique themes: '+str(tedtv1['themes'].nunique()))


# In[47]:


#make a frequency table for all themes
ted_theme=pd.value_counts(tedtv1.themes).to_frame().reset_index()
ted_theme.columns=['Theme','Frequency']
ted_theme


# In[48]:


#select the top 10 themes from the frequency table
ted_theme_top=ted_theme[['Theme', 'Frequency']].sort_values('Frequency', ascending=False)[:10]
print(ted_theme_top)

#make a barplot for the top 10 themes
ted_theme_topplt=ted_theme_top.plot.bar(x='Theme', y='Frequency', xlabel='Themes', ylabel='Popularity', rot=0, figsize=(20,5), title='Most Popular Themes Based', grid=False, legend=False)
ted_theme_topplt


# In[49]:


#based on the mean and the st.dev. we can assume that the best videos must be above 4 mil views
print('View statistics:')
print(round(ted['views'].describe()))


# In[50]:


#selecting the most viewed videos to find popular themes
ted_views_top = tedtv1[tedtv1['views']>=4000000]
print('------------------------------------------')
print('All that are over 4 mil views:')
print(ted_views_top)
print('------------------------------------------')

#make a frequency table for all themes
ted_theme1=pd.value_counts(ted_views_top.themes).to_frame().reset_index()
ted_theme1.columns=['Theme','Frequency']
print('Frequency table:')
print(ted_theme1)
print('------------------------------------------')

#select the top 10 themes from the frequency table
ted_theme1_top=ted_theme1[['Theme', 'Frequency']].sort_values('Frequency', ascending=False)[:10]
print('The top 10 from the Frequency table:')
print(ted_theme1_top)
print('------------------------------------------')

#make a barplot for the top 10 themes
ted_theme1_topplt=ted_theme1_top.plot.bar(x='Theme', y='Frequency', xlabel='Themes', ylabel='Popularity', rot=0, figsize=(20,5), title='Most Popular Themes Based', grid=False, legend=False)
ted_theme1_topplt


# In[66]:


#compare top most frequently used themes with top themes based on views 
pop_theme = set(ted_theme_top['Theme']) & set(ted_theme1_top['Theme'])
print('The most frequenctly seen and most viewed themes:')
print(pop_theme)
print('----------------------------------------------------------------------------------------------')

#plot the common themes via bar charts
fig, az = plt.subplots(2, figsize=(15,8))
fig.suptitle('Best Themes', size=30)
az[0].bar(ted_theme_top['Theme'], ted_theme_top['Frequency'], color=['green', 'green', 'green', 'green', 'green', 'blue', 'green', 'green', 'blue', 'blue', ])
az[0].set_ylabel('Most used Themes', size=20)
az[1].bar(ted_theme1_top['Theme'], ted_theme1_top['Frequency'], color=['green', 'green', 'green', 'green', 'green', 'blue', 'blue', 'green', 'blue', 'green', ])
az[1].set_ylabel('Top views Themes', size=20)
plt.xticks(ted_theme1_top['Theme'])
plt.show()
#the popularity of Health, Innovation, and Design theme is overestimated 
#while the popularity of Psychology, Brain, and Education themes is underestimated 


# ### *Conclusion: Technology, Science, Global issues, Business, Culture, TEDx, Entertainment are the best themes for a TED talk. However, there are overappreciated and underappreciated themes. While Health, Innovation, and Design are overused, Psychology, Brain, and Education are underused by TED talkers.*

# ## Does having a ? or ! in the title matter?

# In[52]:


tednv = ted[['title', 'views']].sort_values('views', ascending = False)
print(tednv)
print('------------------------------------------------------------')

#which type of titles are mostly used?
tednv_q=tednv.loc[tednv['title'].str.endswith('?')]
tednv_e=tednv.loc[tednv['title'].str.endswith('!')]
totqf=len(tednv_e.index)
totef=len(tednv_q.index)
totsf=len(tednv.index)-len(tednv_q.index)-len(tednv_e.index)
print('Based on frequency:')
print('Number of Question titles: '+ str(totqf))
print('Number of Exclamation titles: '+ str(totef))
print('Number of Statement titles: '+ str(totsf))

#which titles types are most successful view generators?
tednv_top = tednv[tednv['views']>=4000000]
tednv_topq=tednv_top.loc[tednv_top['title'].str.endswith('?')]
tednv_tope=tednv_top.loc[tednv_top['title'].str.endswith('!')]
totqv=len(tednv_tope.index)
totev=len(tednv_topq.index)
totsv=len(tednv_top.index)-len(tednv_topq.index)-len(tednv_tope.index)
print('------------------------------------------------------------')
print('Based on the top views:')
print('Number of Question titles: '+ str(totqv))
print('Number of Exclamation titles: '+ str(totev))
print('Number of Statement titles: '+ str(totsv))

#how many of the made titles make it above the 4 mil view level
print('------------------------------------------------------------')
print('Success rate (%):')
print('Number of Question titles: '+ str(round((totqv/totqf)*100,2)))
print('Number of Exclamation titles: '+ str(round((totev/totef)*100,2)))
print('Number of Statement titles: '+ str(round((totsv/totsf)*100,2)))

#the difference between them is not so big, but the best seems to be '!' titles


# ### *Conclusion: The type title (?, !, .) does not matter. But "!" type titles seem to peform a tiny bit better than the rest.*

# ## What is the optimal length of a TED talk title?
# 

# In[53]:


#count all the words for each title row
tednv['count'] = tednv['title'].str.split().str.len()
print(tednv)
print('------------------------------------------------------------')
#Let's do some descriptive analysis for the title legth
print('Title legth statistics:')
print(round(tednv['count'].describe()))
print('------------------------------------------------------------')


# In[54]:


#most fequent title lengths
tednv1=tednv[['views','count']].sort_values('views', ascending=False)
tednv_count=pd.value_counts(tednv1['count']).to_frame().reset_index()
tednv_count.columns=['Count','Frequency']
totcount = tednv_count['Frequency'].sum()
tednv_count['Percentage (%)'] = round((tednv_count.Frequency / totcount)*100, 2)
tednv_count['Cumulative (%)'] = round(100 * (tednv_count['Frequency'].cumsum()/tednv_count['Frequency'].sum()), 2)
tednv_count
# 90% of nost common title legths are 3-9 words long


# In[55]:


#most fequent title lengths
tednv2=tednv1[tednv['views']>=4000000]
tednv_count1=pd.value_counts(tednv2['count']).to_frame().reset_index()
tednv_count1.columns=['Count','Frequency']
totcount1 = tednv_count1['Frequency'].sum()
tednv_count1['Percentage (%)'] = round((tednv_count1.Frequency / totcount1)*100, 2)
tednv_count1['Cumulative (%)'] = round(100 * (tednv_count1['Frequency'].cumsum()/tednv_count1['Frequency'].sum()), 2)
tednv_count1
# almost 90% of most popular title legths are 4-9 words long


# ### *Conclusion: The best title length for a TED talk is from 4-9 words. But most common is 3-9 words. So, the speakers got it right!*

# In[ ]:





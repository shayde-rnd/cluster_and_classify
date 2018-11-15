import urllib
from bs4 import BeautifulSoup

def iterateChildren(parent):
  textList = []
  for child in parent.children:
    if(isinstance(child, str)):
      textList.append(child)
    else:
      if(child.children):
        continue
      textList.append(child.get_text())
  return textList

def extractText(page):
  soup = BeautifulSoup(page)
  content = soup.find("section", id="content")
  span9 = content.find("div", {"class": "span9"})
  textList = iterateChildren(span9)
  return textList

def fetchNewsAndExtractText(id):
  try:
    page = urllib.request.urlopen("http://www.lapdonline.org/october_2018/news_view/{0}".format(id))
    if(page.status == 200):
      return extractText(page)
  except Exception as exp:
    print("Error: {0}".format(exp))
  return None
  
id = 64754
while id > 0:
  newsTextList = fetchNewsAndExtractText(id)
  text = "".join(newsTextList)
  if(text.strip()):
      file = open("{0}.txt".format(id),"w+")
      file.write(text)
      file.close()
  id = id - 1
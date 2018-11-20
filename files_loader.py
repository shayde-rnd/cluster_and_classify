# os can be used for file/dirs manipulations
from os import listdir


def loadFile(path):
  file = open("articles/{0}".format(path), "r")
  text = file.read()
  text = text.decode('utf8').encode('ascii',errors='ignore')
  file.close()
  return text


def loadFiles(dirPath):
  articlesNamesList = listdir(dirPath)
  curpus = []

  for articleName in articlesNamesList:
    #.decode('utf8').encode('ascii',errors='ignore')
    text = loadFile(articleName)
    curpus.append(text)
  return curpus
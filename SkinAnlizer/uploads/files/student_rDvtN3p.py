#-*- coding: utf-8 -*-
import urllib2
from bs4 import BeautifulSoup 


url = "https://ko.wikipedia.org/wiki/%ED%95%B4%EB%8B%AC"
html_doc = urllib2.urlopen(url).read()

def get_soup(url):
    html_doc = urllib2.urlopen(url).read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    return soup

soup = get_soup(url)
HD = soup.find(id="firstHeading" ).string
Sentense = soup.p.get_text()
Img = soup.find("img", alt="Enhydra lutris face.jpg")['src']


print  HD
print  Sentense
print  Img



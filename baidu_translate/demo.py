#/usr/bin/env python
#coding=utf8
 
import httplib
import md5
import urllib
import random
import json
import traceback

appid = '20170805000070810'
secretKey = 'sqt0meDLaM3gmTgZlcT0'
        
class BaiduTranslate():

    def __init__(self):
        self.httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')

    def translate(self, fromLang, toLang, line):
        if len(line)>=2000:
            sents=line.split('.')
            text=''
            texts=[]
            for sent in sents:
                if len(text)<2000:
                    text+=sent
                else:
                    texts.append(text)
                    text=''
            texts.append(text)
        else:
            texts=[line]
        rval=''
        for text in texts:
            tran_text=self._translate(fromLang, toLang, text)
            if not tran_text:continue
            rval+='.'+tran_text
        return rval
            
    def _translate(self, fromLang, toLang, text):
        fromLang = fromLang
        toLang = toLang
        q = text

        myurl = '/api/trans/vip/translate'
        salt = random.randint(32768, 65536)
        sign = appid+q+str(salt)+secretKey
        m1 = md5.new()
        m1.update(sign)
        sign = m1.hexdigest()
        myurl = myurl+'?appid='+appid+'&q='+urllib.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
     
        try:
            self.httpClient.request('GET', myurl) 
            #response是HTTPResponse对象
            response = self.httpClient.getresponse()
            s=response.read()
            #s=eval(s)
            s=json.loads(s)
            return s['trans_result'][0]['dst'].encode('utf-8')
            #return eval('u"'+s['trans_result'][0]['dst']+'"')
        except Exception, e:
            traceback.print_exc()
            return None

nmt=BaiduTranslate()
text='''i bet you 're worried . i was worried . that 's why i began this piece . i was worried about vaginas . i was worried what we think about vaginas , and even more worried that we do n't think about them . i was worried about my own vagina . it needed a context , a culture , a community of other vaginas . there is so much darkness and secrecy surrounding them . like the bermuda triangle , nobody ever reports back from there . in the first place , it 's not so easy to even find your vagina . women go days , weeks , months , without looking at it . i interviewed a high - powered businesswoman ; she told me she did n't have time . " looking at your vagina , " she said , " is a full day 's work . " " you 've got to get down there on your back , in front of a mirror , full - length preferred . you 've got to get in the perfect position , with the perfect light , which then becomes shadowed by the angle you 're at . you 're twisting your head up , arching your back , it 's exhausting . " she was busy ; she did n't have time . so i decided to talk to women about their vaginas . they began as casual vagina interviews , and they turned into vagina monologues . i talked with over 200 women . i talked to older women , younger women , married women , lesbians , single women ; i talked to corporate professionals , college professors , actors , sex workers ; i talked to african - american women , asian - american women , native - american women , caucasian women , jewish women . ok , at first women were a little shy , a little reluctant to talk . once they got going , you could n't stop them . women love to talk about their vaginas -- they do . mainly because no one 's ever asked them before . let 's just start with the word " vagina " -- vagina , vagina . it sounds like an infection at best . maybe a medical instrument . " hurry , nurse , bring the vagina . " vagina , vagina , vagina . it does n't matter how many times you say the word , it never sounds like a word you want to say . it 's a completely ridiculous , totally un - sexy word . if you use it during sex , trying to be politically correct , " darling , would you stroke my vagina , " you kill the act right there . i 'm worried what we call them and do n't call them . in great neck , new york , they call it a " pussy - cat . " a woman told me there , her mother used to tell her , " do n't wear panties , dear , underneath your pajamas . you need to air out your pussy - cat . " in westchester they call it a " pooky , " in'''
text=text*4
ii = open('en.txt')
oo = open('de.txt','w')
for line in ii:
    parts=line.strip().split('\t')
    label=parts[0]
    text=parts[1]
    print 'input',len(text)
    de_text=nmt.translate('en', 'de', text)
    if not de_text:
        continue
    print 'output',len(de_text)
    new_line=label+'\t'+de_text+'\n'
    oo.write(new_line)

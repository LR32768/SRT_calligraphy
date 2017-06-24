# -*-coding: utf-8-*-
import urllib
import urllib.request
import re
import os

from threading import Thread


class TimeoutException(Exception):
    pass


def timelimited(timeout):
    def decorator(function):
        def decorator2(*args, **kwargs):
            class TimeLimited(Thread):
                def __init__(
                        self,
                        _error=None, ):
                    Thread.__init__(self)
                    self._error = _error

                def run(self):
                    try:
                        self.result = function(*args, **kwargs)
                    except Exception as e:
                        self._error = e

                def _stop(self):
                    if self.isAlive():
                        print('Error')

            t = TimeLimited()
            t.start()
            t.join(timeout)

            if isinstance(t._error, TimeoutException):
                t._stop()
                raise TimeoutException('timeout for %s' % (repr(function)))

            if t.isAlive():
                t._stop()
                raise TimeoutException('timeout for %s' % (repr(function)))

            if t._error is None:
                return t.result

        return decorator2

    return decorator


def getHtml(url):
    html = urllib.request.urlopen(url)
    print('loading...')
    print('open' + url)
    page = html.read()
    page = page.decode('utf-8')
    return page


def download(url, name):
    urllib.request.urlretrieve(url, name)


@timelimited(120)
def download_list(urllist, fold=""):
    num = len(urllist)
    try:
        for i in range(num):
            try:
                open(fold + '/%d.jpg' % i)
                print(fold + '/%d.jpg' % i + ' has existed')
            except:
                print('downloading... ' + fold + '/%d.jpg' % i)
                download(urllist[i], fold + '/%d.jpg' % i)
    except TimeoutException:
        os.system('del ' + fold + '/%d.jpg' % i)
        print('reloading...')
        download_list(urllist, fold)


url = 'http://www.sohu.com/a/62891611_389514'

source = getHtml(url)
pattern = re.compile(r'http://photocdn.sohu.com/2016.+jpe?g')
urllist = pattern.findall(source)
download_list(urllist, 'img')

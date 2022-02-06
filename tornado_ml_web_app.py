import tornado.ioloop
import tornado.web
import os
import pandas as pd
import uuid
from tornado.options import define, options
from joblib import dump, load

define("port", default=8888, help="run on the given port", type=int)

def load_model(model_path = 'model/classifier.pkl'):
    path = os.path.abspath(model_path)
    return load(path)  

def predict_model(args):
    iris_classifier = load_model()
    return iris_classifier.predict([args])[0]
    
class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
           (r"/", MainHandler),
           (r"/predict", PredictHandler),
        ]
        settings = dict(
            title=u"Tornado Predictor",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=True,
            cookie_secret= uuid.uuid4().int,
            debug=True,
        )
        super(Application, self).__init__(handlers, **settings)
        
class MainHandler(tornado.web.RequestHandler):
    async def get(self):
        self.render("predict.html", predict=None)
        
class PredictHandler(tornado.web.RequestHandler):
    async def post(self):
        sepal_length = self.get_argument("sepal_length", default=None, strip=False)
        sepal_width = self.get_argument("sepal_width", default=None, strip=False)
        petal_length = self.get_argument("petal_length", default=None, strip=False)
        petal_width = self.get_argument("petal_width", default=None, strip=False)
        try:
            predict = predict_model([sepal_length, sepal_width, petal_length, petal_width])
        except Exception as err:
            self.render("predict.html", predict="Невалидные данные. Попробуйте ещё раз")
        self.render("predict.html", predict=predict)  

if __name__ == "__main__":
    app = Application()
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()

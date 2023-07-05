from flask import Flask, render_template, request
import pickle

from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

#deseralize
clf=pickle.load(open('model.pkl','rb'))


@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])
    features=[int(x) for x in request.form.values()] #maintain the input same as the data that u trained model
    with open('sst.pkl','rb') as file:
        sst=pickle.load(file)

    #sst=StandardScaler()
    #sst=sst.fit()

    output= clf.predict(sst.transform([features]))
    print(output)
    if output[0]==0:
        return render_template("index.html",pred="The Person will not Purchase the SUV")

    else:
        return render_template("index.html", pred="The Person will Purchase the SUV")


if __name__=="__main__":
    app.run(debug=True) #would create a flask local server
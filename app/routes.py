from flask import render_template, flash, redirect, url_for, request, send_file
from app import app, cbir_results
from .forms import SelectImageForm
import json, os, time

def return_value(item):
    temp=0
    for x in cbir_results[3][item]:
        if x[0] == item.split()[0]:
            temp+=1
    return temp
# displays downloads table and forms to interact with downloads
@app.route('/', methods=['GET','POST'])
@app.route('/index', methods=['GET','POST'])
def index():
    # updates values of filters and selected queue for display
    temp = cbir_results[3].keys()
    temp = list(temp)
    temp.sort(key=return_value,reverse=True)
    images = []
    select_image_form = SelectImageForm()
    for item in temp:
        x = item.split()
        path = os.path.join('animal_database',x[0])
        path = os.path.join(path,'original')
        path = os.path.join(path,x[1])
        images.append([path, item])
    return render_template('index.html', title='CBIR',images=images, select_image_form = select_image_form)


# displays a table for files in the gien BASE_DIR (Aria2 download path)
@app.route('/index/image_retrieve', methods=['GET','POST'])
def image_retrieve():
    if request.form['feature-button'] == 'SIFT':
        temp = cbir_results[3][request.form['index']]
        images = []
        x = request.form['index'].split()
        path = os.path.join('animal_database',x[0])
        path = os.path.join(path,'original')
        path = os.path.join(path,x[1])
        query_image = [path, request.form['index']]
        for item in temp:
            path = os.path.join('animal_database',item[0])
            path = os.path.join(path,'original')
            path = os.path.join(path,item[1])
            images.append([path, item[0] + ' ' + item[1]])
        return render_template('image_retrieve.html', title='Retrieved Images', images=images, query_image=query_image)

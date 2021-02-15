import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans,MiniBatchKMeans
import pickle
from statistics import mode
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i])
           #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind


def duplicate_image(image, images):
    for item in images:
        if image.shape != item.shape:
            continue
        else:
            difference = cv2.subtract(image,item)
            b,g,r = cv2.split(difference)
            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                return True
    return False

def Load_Images():
    temp_counter_1 = 0
    temp_counter_2 = 0
    unlabeled_test_images = []
    test_dir = 'app/static/TEST DATA IMAGE/TEST DATA'
    test_files = os.listdir(test_dir)
    for file in test_files:
        if file[-3:] == 'jpg':
            image_path = os.path.join(test_dir, file)
            image = cv2.imread(image_path)
            unlabeled_test_images.append(image)
            temp_counter_1 += 1
    temp_counter_1 = 0
    BASE_DIR = 'app/static/animal_database'
    files = os.listdir(BASE_DIR)
    train_images = {}
    test_images = {}
    for species in files:
        test = []
        train = []
        current_dir = os.path.join(BASE_DIR,species,'original')
        temp = os.listdir(current_dir)
        for file in temp:
            if file[-3:] == 'jpg':
                image_path = os.path.join(current_dir, file)
                image = cv2.imread(image_path)
                if duplicate_image(image, unlabeled_test_images):
                    test.append([image,species,file])
                    temp_counter_1 +=1
                else:
                    temp_counter_2 +=1
                    train.append([image,species,file])
        test_images[species] = test
        train_images[species] = train
    return train_images, test_images

def sift_features(images):
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in images.items():
        features = []
        for img in value:
            kp, des = sift.detectAndCompute(img[0],None)


            descriptor_list.extend(des)
            features.append([des,img[1],img[2]])
        sift_vectors[key] = features
    return descriptor_list, sift_vectors

def kmeans(k, descriptor_list):
    kmeans = MiniBatchKMeans(n_clusters = k, n_init=1)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_
    return visual_words

def image_class(descriptors, centers):
    dict_feature = {}
    for key,value in descriptors.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img[0]:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append([histogram,img[1],img[2]])
        dict_feature[key] = category
    return dict_feature

def knn(images, tests):
    results = {}
    num_test = 0
    correct_predict = 0
    class_based = {}
    all_train = []
    for train_key, train_val in images.items():
        all_train = all_train + train_val


    for test_key, test_val in tests.items():
        for tst in test_val:
            all_train.sort(key=lambda item : distance.euclidean(item[0],tst[0]))
            pred = []
            closest = []
            for item in all_train[:9]:
                closest.append([item[1],item[2]])
                pred.append(item[1])
            pred = mode(pred)
            if test_key not in class_based.keys():
                class_based[test_key] = [0,0]
            if pred == test_key:
                class_based[test_key] = [class_based[test_key][0]+1,class_based[test_key][1]+1]
                num_test += 1
                correct_predict += 1
            else:
                class_based[test_key] = [class_based[test_key][0],class_based[test_key][1]+1]
                num_test += 1
            results[tst[1] + ' ' + tst[2]] = closest

    return [num_test, correct_predict, class_based, results]


# Calculates the average accuracy and class based accuracies.
def accuracy(results):
    relative_class = {}
    relative_all = 0
    for item in results[3].keys():
        if item.split()[0] not in relative_class:
            relative_class[item.split()[0]] = 0
        for pre in results[3][item]:
            if pre[0] == item.split()[0]:
                relative_class[pre[0]] = relative_class[pre[0]]+1
                relative_all +=1

    avg_accuracy = (results[1] / results[0]) * 100
    print("Average accuracy: %" + str(avg_accuracy) , relative_all/results[0])
    print("\nClass based accuracies: \n")
    for key,value in results[2].items():
        acc = (value[0] / value[1]) * 100
        print(key + " : %" + str(acc) , relative_class[key]/value[1])

def CBIR(number_of_vw):
    for item in number_of_vw:
        number_of_features = item
        objects = os.listdir('obj')
        descriptor_list = 0
        train_sift_features = 0
        test_sift_features = 0
        visual_words = 0
        bovw_test = 0
        bovw_train = 0
        results_bowl = 0
        if ('results_sift'+str(number_of_features)+'.pkl') not in objects:
            if ('bovw_train'+str(number_of_features)+'.pkl') not in objects or ('bovw_test'+str(number_of_features)+'.pkl') not in objects:
                if 'descriptor_list.pkl' not in objects:
                    descriptor_list, train_sift_features = sift_features(train_images)
                    _,test_sift_features = sift_features(test_images)
                    save_obj(test_sift_features, ('test_sift_features'))
                    save_obj(train_sift_features, ('train_sift_features'))
                    save_obj(descriptor_list, ('descriptor_list'))
                    print('sift extracted')
                elif(descriptor_list == 0):
                    descriptor_list = load_obj('descriptor_list')
                    test_sift_features = load_obj('test_sift_features')
                    train_sift_features = load_obj('train_sift_features')
                    print('features loaded')
                # Takes the central points which is visual words
                if ('visual_words'+str(number_of_features)+'.pkl') not in objects:
                    visual_words = kmeans(number_of_features, descriptor_list)
                    save_obj(visual_words, ('visual_words'+str(number_of_features)))
                    print('visual words extracted' )
                else:
                    visual_words = load_obj('visual_words'+str(number_of_features))
                    print('visual words loaded')
                if ('bovw_train'+str(number_of_features)+'.pkl') not in objects:
                    bovw_train = image_class(train_sift_features, visual_words)
                    save_obj(bovw_train, ('bovw_train'+str(number_of_features)))
                    print('train histogram created')
                else:
                    bovw_train = load_obj('bovw_train'+str(number_of_features))
                    print('train histogram loaded')
                if ('bovw_test'+str(number_of_features)+'.pkl') not in objects:
                    bovw_test = image_class(test_sift_features, visual_words)
                    save_obj(bovw_test, ('bovw_test'+str(number_of_features)))
                    print('test histogram created')
                else:
                    bovw_test = load_obj('bovw_test'+str(number_of_features))
                    print('test histogram loaded')
            else:
                bovw_test = load_obj('bovw_test'+str(number_of_features))
                bovw_train = load_obj('bovw_train'+str(number_of_features))
                print('histograms loaded')
            # Call the knn function
            results_bowl = knn(bovw_train, bovw_test)
            save_obj(results_bowl, ('results_sift'+str(number_of_features)))
            print('results saved')
        else:
            results_bowl = load_obj(('results_sift'+str(number_of_features)))
            print('results loaded')

        print(number_of_features)
        # Calculates the accuracies and write the results to the console.
        accuracy(results_bowl)

if __name__ == '__main__':
    #train_images, test_images = Load_Images()
    number_of_vw = [200,220,500,1000]
    CBIR(number_of_vw)

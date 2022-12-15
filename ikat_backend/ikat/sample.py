#     #scene detection
    #     lst_mean = []
    #     lst_centers = []
    #     feat = []
    #     features = []
    #     features1 = []
    #     lst_centers1 = []
    #     lst_distances = []


    #     path = "./media"
    #     result = glob.glob(path+'/*.jpg')
    #     for j,path1 in enumerate(result):
    #         image = cv2.imread(path1)
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #         #image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            
    #         mean  = scipy.mean(image,axis=0)
            

            
    #         mean = mean.tolist()
    #         std = scipy.std(image,axis=0)
    #         std = std.tolist()
            

    #         skewness = stats.skew(image,axis=0)
    #         skewness = skewness.tolist()
    #         skewness = np.concatenate((mean,std,skewness),axis= 1)
    #         norm_mom = np.linalg.norm(skewness)
    #         features1.append(norm_mom)





    #         hue = image[0]
    #         saturation  = image[1]
    #         meana = scipy.mean(hue,axis = 0)
            

    #         stda= scipy.std(hue,axis = 0)
    #         #print(std)
            

    #         mean1 = scipy.mean(saturation,axis = 0)
    #         #print(mean1)
            

    #         std1 = scipy.std(saturation,axis = 0)
    #         #print(std1)


    #         con = np.concatenate((meana,stda,mean1,std1),axis = 0)
    #         #print(con)
    #         norm_con = np.linalg.norm(con)
    #         features1.append(norm_con)
    #         image  = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    #         can = cv2.Canny(image,100,200)
    #         can = np.array(can)
    #         norm_can = np.linalg.norm(can)

    #         features1.append(norm_can)

    #         gabor  = filters.frangi(image)
    #         image = image[:,:,:]
    #         print(gabor.shape)
    #         df = pd.DataFrame()
    #         num = 1
        
    #         for i in range(1,6):
    #             i = (i/4)*3.14
    #             for h in range(3,9):
    #                 for j in range(1,4):
    #                     for k in range(1,2):

    #                         kernel = cv2.getGaborKernel((39,39),h,j,i,k,0,ktype=cv2.CV_32F)
    #                         img  = cv2.filter2D(image,cv2.CV_8UC3,kernel)
    #                         img = img.reshape(-1)
    #                 #	df[Gabor] = img
    #                 #	df.to_csv("Gabor.csv")
    #                         num = num + 1
    #         # print(img)
    #         gab = np.array(img)
    #         gab_norm = np.linalg.norm(gab)
    #         features1.append(gab_norm)
        
            
    #         image = image[:,:,0]
    #         mat = feature.graycomatrix(image,[1],[45])
    #         tamura = feature.graycoprops(mat,prop = 'contrast')
    #         tamura1 = feature.graycoprops(mat,prop = 'dissimilarity')
    #         tamura2 = feature.graycoprops(mat,prop = 'homogeneity')
    #         #with open("chromatic.txt",'a') as g:
    #             #g.write(str(con))
    #             #g.write('\n')


    #         arr1= np.concatenate((tamura,tamura1,tamura2),axis = 1)
    #         #print(arr.shape)
    #         #arr= arr.reshape(arr[0],arr[1]*arr[2])

    #         #arr = arr.flat()
    #         #print(dir(arr))
    #         #arr = features.tolist()
    #         norm_tam = np.linalg.norm(arr1)
    #         #print(norm)


            
    #         #print(arr)
            
    #         #print(arr.size)
            
    #         #print(arr)
    #         features1.append(norm_tam)
    #     #print(feat)
    #         features2  = np.array(features1)
    #         features_mean = np.mean(features2)
    #     print(features_mean)
    #     with open("features_mean.txt",'a') as h:


    #         h.write(str(features_mean))
    #         h.write('\n')
    #     with open("features_mean.txt",'r') as k:
    #         for l in k:

    #         #l= np.array(l)
    #     # l= pd.to_numeric(l)
    #     #l = np.concatenate(l)
    #         #l= pd.to_numeric(l)
    #             lst_mean.append(l)
    #     lst = [ast.literal_eval(a) for a in lst_mean]
    #     print(lst)
    #     module_dir = os.path.dirname(__file__)  
    #     file_path = os.path.join(module_dir, 'centers.txt')   #full path to text.
    #     #data_file = open(file_path , 'r')       
    #     #data = data_file.read()

    #     with open(file_path,'r') as g:



    #         for o in g:
    #             #o = np.array(o)
    #             #o=  o.tolist()
    #             o = o.replace('[','')
    #             o = o.replace(']','')
    #             o = o.strip()
    #             #o = pd.to_numeric(o)
    #             #dist = np.subtract(l,o)
    #             lst_centers.append(o.strip())
            

    #     print(lst_centers)
    #     #for u in lst_centers:
    #     #u = u.replace("'"," ")
    #     #lst_centers1.append(u)
    #     #print(lst_centers1)
    #     lst2 = [ast.literal_eval(t) for t in lst_centers]
    #     print(lst2)
    #     for d in lst:
    #         for y in lst2:
    #             dist = np.subtract(d,y)
    #             dist =   np.linalg.norm(dist)
    #             lst_distances.append(dist)
    #             #sorted = np.sort(lst_distances)
    #     minimum = np.min(lst_distances)
    #     index = lst_distances.index(minimum)
    #     print("SCENE BELONGS TO CLUSTER NUMBER"+str(index))
	
	

    # #dis1 = np.array(dist)
    # #dist = dist.tolist()

    #     #Attribute Selection
    #     #path = "./media"
    #     #$weights =  models.ResNet50_Weights.DEFAULT
    #     model = models.vgg16(pretrained = True)

    #     model = model.type(torch.cuda.FloatTensor)
    #     print(summary(model,(3,256,256)))

    #     class extract(nn.Module):
    #         def __init__(self,model):
    #             super(extract,self).__init__()
    #             self.features  = list(model.features)
    #             self.features = nn.Sequential(*self.features)
    #             self.pooling = model.avgpool
    #             self.flatten = nn.Flatten()
    #             #self.linear = nn.Linear(1,10)
    #             self.fc = model.classifier[1]

    #         def forward(self,input):
    #             out = self.features(input)
    #             out = self.pooling(out)
    #             out = self.flatten(out)
    #             out = self.fc(out)
    #             return out 
    #     #model = models.resnet50(weights = weights)

    #     updated = extract(model)
    #     print(summary(updated,(3,256,256)))

    #     transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    #     features3 = []
    #     result  = glob.glob(path+"/*.jpg")
    #     for i,path1 in enumerate(result):
    #         image = cv2.imread(path1)
    #         image = transform(image)
    #         image = image.type(torch.cuda.FloatTensor)
    #         image = image.unsqueeze(0)
    #     #image = image.to(device)
    #         #print(type(image))
    #     #feature = updated(image)
    #         with torch.no_grad():
    #             feature4 = updated(image)
    #         features3.append(feature4.cpu().detach().numpy().reshape(-1))
    #     features3 = np.array(features3)
    #     #with open("/content/drive/MyDrive/SIH/features"+str(i)+".txt",'a') as f:

    #      #   f.write(str(features3))
    #       #  f.write("\n")

    #     print(features3)



    #     #segmentation
    #     ocrWIN = np.ones([50,500])

    #     tr.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    #     #cap = cv2.VideoCapture(0)

    #     ins = instanceSegmentation()
    #     #module_dir = os.path.dirname(__file__)  
    #     model_path = os.path.join(module_dir, 'pointrend_resnet50.pkl')
    #     ins.load_model(model_path,detection_speed = "rapid")

    #     t = " "

    #     ins.segmentImage(image_seg, show_bboxes=True, output_image_name="output_image.jpg")
    #     #image2 = cv2.imwrite("output_image.jpg",image1)

    #     img = cv2.imread("output_image.jpg")
    #     #imgG = frame.copy()
    #     imgG = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #     module_dir = os.path.dirname(__file__)  
    #     img_read= os.path.join(module_dir, 'img.png') 
    #     imgOCR = cv2.imread(img_read)
    #     imgOCR = cv2.cvtColor(imgOCR,cv2.COLOR_BGR2RGB)

    #     #imgOCR = frame.copy()
    #     #imgOCR = cv2.cvtColor(imgG,cv2.COLOR_BGR2RGB)
    #         # t = " "
    #     t = tr.image_to_string(imgOCR)
    #     edges = cv2.Canny(imgG,100,200)

    #     cv2.putText(img = ocrWIN,
    #         text = t,
    #         org = (50, 15),
    #         fontFace = cv2.FONT_HERSHEY_DUPLEX,
    #         fontScale = 0.6,
    #         color = (0, 0, 0),
    #         thickness = 1
    #     )

    #     print(t)
    #     # a=cv2.imshow("Segmentation",img)
    #     # b=cv2.imshow("OCR",ocrWIN)
    #     # c=cv2.imshow("Boundary",edges)
    #     #cv2.imwrite("Segmentation.png",img)
    #     #cv2.imwrite("OCR.png",ocrWIN)
    #     #cv2.imwrite("Boundary.png",edges)
    #     cv2.imwrite(settings.MEDIA_ROOT +"/Segmentation.png",img)
    #     cv2.imwrite(settings.MEDIA_ROOT+"/OCR.png",ocrWIN)
    #     cv2.imwrite(settings.MEDIA_ROOT +"/Boundary.png",edges)




    #     img_1=ResultImage()
    #     img_1.image="../media/Segmentation.png"
    #     img_1.save()
    #     print("img_1 saved")
    #     img_2=ResultImage()
    #     img_2.image="../media/OCR.png"
    #     img_2.save()
    #     print("img2_save")
    #     img_3=ResultImage()
    #     img_3.image="../media/Boundary.png"
    #     img_3.save()
    #     # print("img3 saved")
    #     # Employee.objects.create(
    #     # first_name='Bobby',
    #     # last_name='Tables'
    #     # )
    #     # cv2.waitKey(1)
    #     # cv2.destroyAllWindows()
    #     #cv2.imshow("Boundary",edges)
    #     #cv2.imshow("OCR",ocrWIN)
    #     # ocrWIN = np.ones([50,500])
    #     print("reached here")
    #     ri = ResultImage.objects.all()



    # context = {
    #     'filePathName':filePathName,
    #     'centers':centers,
    #     'features':features3,
    #     'minimum':minimum,
    #     'index':index,
    #     't':t,
    #     'ri':ri,
    # }

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import layoutparser as lp
import pypdfium2 as pdfium
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTChar, LTTextLine
import subprocess
import os
from PIL import Image, ImageEnhance
import json
import re
import pickle


class pdf_table_parser:

    def __init__(self, file_name, model, output_dir="data/output", input_dir="data/input", origin_dir="", paddle_dir="PaddleOCR", page_resize_factor=2, table_resize_factor=2, enhance_factor=2):

        self.file_name = file_name
        self.model = model
        self.input_dir = os.path.join(origin_dir,input_dir)
        self.output_dir = os.path.join(origin_dir,output_dir)
        self.paddle_dir = os.path.join(origin_dir,paddle_dir)
        self.page_resize_factor = page_resize_factor
        self.table_resize_factor = table_resize_factor
        self.enhance_factor = enhance_factor

        self.report_path = os.path.join(self.input_dir,file_name)

        self.page_layout_list = list(extract_pages(self.report_path))
        self.pdf_obj  = pdfium.PdfDocument(self.report_path)
        self.n_pages = len(self.pdf_obj)
        self.make_output_dir()
        
    
    def make_output_dir(self):
        report_name = self.file_name.split(".")[0]
        self.store_path = os.path.join(self.output_dir,report_name)
        self.images_path = os.path.join(self.store_path,"images")
        self.output_path = os.path.join(self.store_path,"output")
        if(not os.path.exists(self.store_path)):
            os.mkdir(self.store_path)
        if(not os.path.exists(self.images_path)):
            os.mkdir(self.images_path)
        if(not os.path.exists(self.output_path)):
            os.mkdir(self.output_path)


    def get_number_of_pages(self):
        return self.n_pages

    def display_page(self,page_number):
        page_pdf = self.pdf_obj.get_page(page_number-1)
        page_image = page_pdf.render().to_pil()
        page_image.show()
        

    def enhance_image(self, image, resize_factor, enhance_factor=None):

        if(enhance_factor==None):
            enhance_factor = self.enhance_factor

        image = image.convert('LA')
        enhancer = ImageEnhance.Contrast(image)
        img_enhance = enhancer.enhance(enhance_factor)
        img_dim = np.array(img_enhance).shape
        img_resized = img_enhance.resize((img_dim[1]*resize_factor, img_dim[0]*resize_factor), Image.LANCZOS)

        return img_resized

    def clean_text(self,text):
        """ 
        Replace /n with <br>
        """
        text = text.replace('\n','<br>')
        return text
    

    def save_table_image(self, table_image, page_number, table_index):
        table_file_name = str(page_number)+"_"+str(table_index)+".png"
        table_file_path = os.path.join(self.images_path,table_file_name)
        print(table_file_path)
        table_image.save(table_file_path)
        return table_file_path
        

    def get_bbox(self,image, threshold=0.4):
        layout = self.model.detect(image)

        bbox_list = []
        conf_score_list = []

        for block in layout:
            if(block.type=='Table' and block.score>threshold):
                start_point = (int(block.block.x_1),int(block.block.y_1))
                end_point = (int(block.block.x_2), int(block.block.y_2))
                
                bbox_list.append((start_point, end_point))
                conf_score_list.append(block.score)
        return bbox_list, conf_score_list

    def bbox_within_bbox(self,bbox_element, bbox_table):
        """
        Returns whether the given pdf element lies inside the bounding box of the table
        """
        if( bbox_element[1]<=bbox_table[1][1] and bbox_element[3]>=bbox_table[0][1]  and bbox_element[0]<=bbox_table[1][0] and bbox_element[2]>=bbox_table[0][0]):
            return True
        else:
            return False

    
    def get_table_image_elements(self, page_elements, page_image, bbox):
         
        #Returns list of word with bbox
        

        cropped_image = page_image.crop((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))
        cropped_image_enhance = self.enhance_image(cropped_image , resize_factor=self.table_resize_factor)
        #cropped_image_enhance.show()

        page_bbox = page_elements.bbox      #gives the bounding box of the entire page
        data_list = []
        for element in page_elements:
            elem_bbox = element.bbox
            elem_bbox_mod = [elem_bbox[0], page_bbox[-1]-elem_bbox[3], elem_bbox[2], page_bbox[-1]-elem_bbox[1]]

            if(self.bbox_within_bbox(elem_bbox_mod, bbox) and isinstance(element, LTTextBox)):
                for line in element:
                    word = ""
                    word_bbox = None
                    size = None
                    for char in line:
                        if(isinstance(char, LTChar) and char._text!=" "): 
                            word += char._text
                            if(word_bbox==None):
                                word_bbox= char.bbox
                            else:
                                word_bbox = [min(char.bbox[0], word_bbox[0]), word_bbox[1], max(char.bbox[2], word_bbox[2]), word_bbox[3]]
                    
                            if(size==None):
                                size = char.size
                        else:
                            if(len(word)!=0):
                                data_list.append((word_bbox, self.clean_text(word)))
                                word=""
                                word_bbox=None
        
        return cropped_image_enhance, data_list
    
    """    
    def get_table_image_elements(self, page_elements, page_image, bbox):

        cropped_image = page_image.crop((bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))
        cropped_image_enhance = self.enhance_image(cropped_image , resize_factor=self.table_resize_factor)
        #cropped_image_enhance.show()

        page_bbox = page_elements.bbox      #gives the bounding box of the entire page
        data_list = []
        for element in page_elements:
            elem_bbox = element.bbox
            elem_bbox_mod = [elem_bbox[0], page_bbox[-1]-elem_bbox[3], elem_bbox[2], page_bbox[-1]-elem_bbox[1]]
            if(self.bbox_within_bbox(elem_bbox_mod, bbox) and isinstance(element, LTTextBox)):
                text = self.clean_text(element.get_text())

                #text = element.get_text()
                #print(text)
                data_list.append((element.bbox, text))
        
        return cropped_image_enhance, data_list
    
    """
    
    def mapping_table_elements(self,table_elements, page_bbox, image_bbox, resize_factor):

        for i,element in enumerate(table_elements):
            bbox = element[0]

            #Convert from pdf coordinate to image coordinate
            bbox = (
                (bbox[0] - image_bbox[0][0])*resize_factor,
                (page_bbox[-1] - bbox[3] - image_bbox[0][1])*resize_factor,
                (bbox[2] - image_bbox[0][0])*resize_factor,
                (page_bbox[-1] - bbox[1] - image_bbox[0][1])*resize_factor
            )

            table_elements[i] = (bbox, element[1])

        return table_elements

    def get_table_structure(self, table_image_path):
        
        subprocess.run([
            "python3",
            os.path.join(self.paddle_dir,"ppstructure/table/predict_structure.py"),
            "--table_model_dir="+os.path.join(self.paddle_dir,"ppstructure/inference/en_ppstructure_mobile_v2.0_SLANet_infer"),
            "--table_char_dict_path="+os.path.join(self.paddle_dir,"ppocr/utils/dict/table_structure_dict.txt"),
            "--image_dir="+table_image_path,
            "--output="+self.output_path
        ])

        with open(os.path.join(self.output_path,'infer.txt'),'r') as f:
            s = f.readline()[:-1]

        b_1 = s.index('[')
        b_2 = s.index(']')

        b_3 = b_2+3
        html_tags_str = s[b_1:b_2+1]
        html_tags_coord_list = json.loads(s[b_3:])
        html_tags_list = eval(html_tags_str)        #eval converts the list in string format to list format
        #html_tags_insert_indices = [index.start()+4 for index in re.finditer(pattern='<td></td>', string=html_tags_str)]
        
        return html_tags_list, html_tags_coord_list
        
    def measure_offset(self,bbox_1,bbox_2):
        h1 = bbox_1[3] - bbox_1[1]
        h2 = bbox_2[3] - bbox_2[1]

        w1 = bbox_1[2] - bbox_1[0]
        w2 = bbox_2[2] - bbox_2[0]

        h = 2*h1*h2/(h1+h2)
        w = 2*w1*w2/(w1+w2)

        l_err = abs(bbox_1[0]-bbox_2[0])/w
        r_err = abs(bbox_1[2]-bbox_2[2])/h
        u_err = abs(bbox_1[1]-bbox_2[1])/w
        d_err = abs(bbox_1[3]-bbox_2[3])/h

        err = (l_err,r_err,u_err,d_err)
        max_err = max(err)
        sum_err = sum(err)
        return sum_err

    def overlap(self, bbox_1, bbox_2):
        area_bbox_1 = (bbox_1[2] - bbox_1[0]) * (bbox_1[3] - bbox_1[1])
        area_bbox_2 = (bbox_2[2] - bbox_2[0]) * (bbox_2[3] - bbox_2[1])

        sum_area = area_bbox_1 + area_bbox_2

        left_line = max(bbox_1[0], bbox_2[0])
        right_line = min(bbox_1[2], bbox_2[2])
        top_line = max(bbox_1[1], bbox_2[1])
        bottom_line = min(bbox_1[3], bbox_2[3])

        if left_line >= right_line or top_line >= bottom_line:
            return 0.0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


    def get_tables(self):
        
        key_base = self.file_name.split(".")[0]
        tables_dict = {}
        
        for i in range(self.n_pages):
            page_pdf = self.pdf_obj.get_page(i)
            page_image = page_pdf.render().to_pil()
            page_elements = self.page_layout_list[i]

            bbox_list, conf_score_list = self.get_bbox(page_image)

            print("Page ",(i+1)," - Number of tables ",len(bbox_list))
            for j,bbox in enumerate(bbox_list):

                #print(bbox)
                table_image, table_elements = self.get_table_image_elements(
                    page_elements = page_elements,
                    page_image = page_image,
                    bbox=bbox
                )

                #Mapping table elements onto the table image
                table_elements = self.mapping_table_elements(
                    table_elements=table_elements,
                    page_bbox = page_elements.bbox,
                    image_bbox = bbox,
                    resize_factor = self.table_resize_factor
                )

                table_image_path = self.save_table_image(
                    table_image=table_image,
                    page_number = (i+1),
                    table_index = j
                )

                html_tags_list, html_tags_coord_list = self.get_table_structure(table_image_path)

                html_tags_insert_indices = []

                for k,tag in enumerate(html_tags_list):
                    if('</td>' in tag):
                        html_tags_insert_indices.append(k)

                acc_mapping = 0     #Count the number of elements mapped below a threshold
                for element in table_elements:
                    bbox = element[0]
                    text = element[1]


                    relative_position = [
                        (html_tags_insert_indices[x], 
                        1 - self.overlap(bbox,html_tags_coord_list[x]), 
                        self.measure_offset(bbox,html_tags_coord_list[x]))
                        for x in range(len(html_tags_coord_list))   
                    ]

                    sorted_relative_position = relative_position.copy()
                    sorted_relative_position = sorted(
                        sorted_relative_position,
                        key=lambda item:(item[1], item[2])
                    )

                    #arg_min_offset_array = np.argmin(offset_array)
                    #sum_err = offset_array[arg_min_offset_array]
                    #print(text,sum_err)

                    """
                    if(sum_err<2):
                        acc_mapping += 1
                    """

                    insert_index = sorted_relative_position[0][0]
                    html_tags_list[insert_index] = html_tags_list[insert_index][:-5] + text + " " + html_tags_list[insert_index][-5:]

                style_tag = '<head><style>table {border-collapse: collapse;border: 1px solid black;}tr {border-bottom: 1px solid black;border-top: 1px solid black;}td {border-left: 1px solid black;border-right: 1px solid black;}</style></head>'
                html_tags_list = html_tags_list[:2] + [style_tag] + html_tags_list[2:]

                acc_percent = None
                if(len(table_elements)!=0):
                    acc_percent = acc_mapping*100/len(table_elements)

                print("Page ",(i+1),"Table ",j," accuracy : ",acc_percent)

                table_html = " ".join(html_tags_list)
                with open(os.path.join(self.output_path,str(i+1)+"_"+str(j)+".html"),'w') as f:
                    f.write(table_html)

                key = key_base + "_" + str(i+1) + "_"+str(j)
                tables_dict[key] = {
                    "table":table_html,
                    "score":acc_percent
                }
        
        return tables_dict
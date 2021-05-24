# Online Inference for model (homework2)
___________
**Prerequisites**
~~~
1. pip (version >= 20.0.2)
2. python (version >= 3.6.7)
3. pytest (version >= 6.2.3)
4. docker (version >= 5.0.0)
~~~
**Instalation**
* **From source**
~~~
git init
git clone https://github.com/made-ml-in-prod-2021/alinazemlev.git
git checkout homework2
cd online_inference
docker build -t alinazemlev/online_inference:v1
~~~
* **From DockerHub**
~~~
docker pull alinazemlev/online_inference:v1
~~~
**Usage**
* **Run serves (First terminal)**
~~~
docker run -p 80:80 alinazemlev/online_inference:v1
~~~
* **Run requests (Second terminal)**
~~~
python make_request.py
~~~
**Tests**
~~~
pytest . -v
~~~
Project Organization
------------

    
    
    ├── online_inference        <- Source code and data for use in this project.
    │   ├── tests           
    │   │   └── test_app.py     <- Tests for the app
    │   │   └── __init__.py     <- Makes src a Python module
    │   │  
    │   ├── README.md           <- The top-level README for developers using this project.
    │   │     
    │   ├── __init__.py         <- Makes src a Python module
    │   │
    │   ├── app.py              <- Source code for FastAPI aplication 
    │   │
    │   ├── make_request.py     <- Code with requests to the app
    │   │
    │   ├── pipeline.py         <- Source code with pipeline which creates the features and the model
    │   │
    │   ├── model.pkl           <- Trained pipeline
    │   │
    │   ├── Dockerfile          <- A text document that contains in order, needed to build an image
    │   │
    │   └── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`        
    │
    └──
--------
**Самоценка**
1. Inference модели обернут в rest сервис FastApi. Есть endpoint /predict - 3 балла
2. Есть тест для /predict - 3 балла
3. Есть скрипт, который делает запросы к сервису - 2 балла
4. Есть валидация входных данных. В случае если список колонок не совпадает с данными обучения возникает ошибка 400 - 3 балла
5. Есть dockerfile, написана в readme корректная команда сборки -4 балла
6. Опубликован образ  - 2 балла
7. Написаны в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference модель  - 1 балл.
8. Самооценка - 1 балл
--------
Итого: 19 баллов.




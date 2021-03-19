使用LSTM  
將2019一月~2020十二月   
全部的尖峰供電、負載、備轉容量、備轉容量率、工業用電以及民生用電整合  
以前面200天預測之後的60天  
最後再output其中7天的資訊  
目前是以2021年的1/1到1/30做為測試資料   
![image](https://user-images.githubusercontent.com/66662065/111767372-9abc8f80-88e1-11eb-91c3-f194ed59b4d3.png)

![image](https://user-images.githubusercontent.com/66662065/111776415-f2142d00-88ec-11eb-9bf6-b3a756f9d422.png)


loss:0.0072, vl_loss: 0.0084  RMSE:304.9612084582683 (1/31-2/6)   
python app.py --training 2021年test.csv --output submission.csv      
---------------------------------------------------------------------[20210312]   
考慮到近期的資訊只有備轉容量與備轉容量率      
新增另一個LSTM

![image](https://user-images.githubusercontent.com/66662065/111767403-a445f780-88e1-11eb-8250-6a31577684c0.png)

以前面60天預測後7天
將其與前面model合併      
預測出的兩筆資料前多乘上一個參數整合   
python app.py --training1 2021年test.csv --training2 J2021年test.csv --output submission.csv  

![image](https://user-images.githubusercontent.com/66662065/111773972-cd6a8600-88e9-11eb-8f0e-607580b06d17.png)

RMSE:110.33170284218629 (3/9-3/15)      
---------------------------------------------------------------------[20210319] 

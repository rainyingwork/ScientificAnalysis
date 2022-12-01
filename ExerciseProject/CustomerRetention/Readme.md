參考網站
1.ithome 11th鐵人賽 I code so I am 一服見效的 AI 應用系列 
https://ithelp.ithome.com.tw/articles/10215639

使用各種集群分析(Clustering)的演算法，將客戶自動分群，進而找出 VIP 客戶。

加強
實務上我們可以做得更精緻一點，例如：

可以分地區，每個地區個別作RFM，因為，各地區消費習慣可能不同，或者行銷活動的目標客群只鎖定某一區域。

篩選出來的客戶數，可經由K值控制，若篩選出來的客戶數太多，可以加大K值，或改變RFM的分類。

可以使用更精良的演算法，例如GMM(Gaussian Mixture Model)等，k-means 使用歐幾里得距離，通常使集群作正圓形的切分，而GMM可以依各種形狀切分，亦即GMM會考慮依各特徵離散程度作調整。
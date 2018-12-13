# ----------------导入相关package-----------------
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

import torch.utils.data as Data

def fourier_transformation (input_data,rows,frequency):
   
    #将输入数据进行傅里叶变换
    transformed = np.fft.fft(input_data[:,rows])

    #将输入数据指定频率以上滤除
    transformed [ np.real(transformed) < frequency] = 0

    #将数据进行傅里叶反变换
    transformed_back = np.fft.ifft(transformed)

    #将反变换之后的数据重新
    input_data [:,rows] = transformed_back

    #返回原始数据
    return input_data

if __name__ == '__main__':

    torch.manual_seed(1)    # reproducible

    ## --------------------载入数据---------------------

    # 定义数据量多少
    data_size = 15000

    # ----------------2.速度信息--------------

    # 将数据转化成载入并转化成numpy
    data_odom = np.loadtxt('odom.txt',dtype= 'str',skiprows=0,delimiter=",")


    #只筛选特定的行
    #data_odom_extratct = data_odom[1:20, [0,48,49,50,51,52,53]]
    data_odom_extratct = data_odom[ : , [0,5,6,7,8,9,10,11,48,49,50,51,52,53]]

    # 针对平面问题的数据筛选

    #只筛选特定的行
    
    #data_odom_extratct = data_odom_extratct[ : , [1,2,6,8,9,13]]

    data_odom_extratct = data_odom_extratct[ : , [1,2,6,8,13]]



    # --------------------将数据拼接在一个numpy当中---------------------

    # 取数据前2000行
    data_odom_extratct_data_size = data_odom_extratct[ 1:data_size+1 , :]

    print("data_odom_extratct_data_size")
    print(data_odom_extratct_data_size[0:20,:])

    #生成输入数据 
    data_input = data_odom_extratct_data_size.astype(np.float)

    print("data_input")
    print(data_input[1:20,:])

    # 定义乘数
    data_input[:, 2] = data_input[:, 2] *10
    #data_input[:, 3] = data_input[:, 3] *1000
    data_input[:, 4] = data_input[:, 4] *10
    #data_input[:, 5] = data_input[:, 5] *1000

    data_input = data_input

    # 生成输出数据 
    data_output = data_odom_extratct[2:data_size+2 ,[0,1,2] ].astype(np.float)

    # 定义乘数
    data_output[:, 2] = data_output[:, 2] *10

    data_output = data_output

    # ------------将数据进行傅里叶变换-----------
    #data_input = fourier_transformation(data_input,3,20)
    #data_input = fourier_transformation(data_input,4,20)

    #X = np.linspace(1, data_size, data_size, endpoint=True)
    #plt.plot(X,data_input[:,4])
    #plt.show()


    # ------------将numpy转化成tensor-----------

    data_input_torch = torch.from_numpy(data_input)
    data_input_float = data_input_torch.float() # 转化成浮点数

    data_output_torch = torch.from_numpy(data_output)
    data_output_float = data_output_torch.float() # 转化成浮点数 

    print("data_input_torch")
    print(data_input_torch[1000:1200,:]) 
    print(data_input_torch.shape)


    print("data_output_torch")
    print(data_output_torch[1000:1200,:]) 
    print(data_output_torch.shape)

    # ------------进行批训练-----------

    BATCH_SIZE = 2

    # 定义数据库 （输入输出分别是之前的输入输出）
    dataset = Data.TensorDataset(data_input_float, data_output_float) 

    # 定义数据加载器
    loader = Data.DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)


    #----------------定义相关网络-----------------

    # 定义迭代次数
    times = data_size

    # 生成随机输出变量
    data_plot = torch.zeros(times,3) #设定了一千条数据

    # 生成损失函数误差变量
    loss_plot = torch.zeros(times,3) #设定了一千条数据

    # 首先，定义所有层属性
    class Net(torch.nn.Module):  # 继承 torch 的 Module
    
        #定义该神经网络：4个全连接层，每层元素128个
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()     # 继承 __init__ 功能
            # 定义每层用什么样的形式
            self.fc1 = torch.nn.Linear(n_feature, n_hidden)   # 第一个全连接层
            self.fc2 = torch.nn.Linear(n_hidden, n_hidden)   # 第二个全连接层
            self.fc3 = torch.nn.Linear(n_hidden, n_hidden)   # 第三个全连接层
            self.fc4 = torch.nn.Linear(n_hidden, n_output)   # 第四个全连接层
            #self.dropout = torch.nn.Dropout(p=0.5)
    
        #定义前向网络
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            #x = self.dropout(x)
            return x

        # #定义前向网络
        #def forward(self, x):
        #    x = F.sigmoid(self.fc1(x))
        #    x = F.sigmoid(self.fc2(x))
        #    x = F.sigmoid(self.fc3(x))
        #    x = self.fc4(x)
        #    x = self.dropout(x)
        #    return x

    net = Net(n_feature=5, n_hidden=128, n_output=3) 

    print(net)


    #----------------定义优化方法&定义损失函数-----------------

    #使用“随机梯度下降法”进行参数优化
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)  # 传入 net 的所有参数, 学习率

    #使用“ADAM”进行参数优化
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001) # 传入 net 的所有参数, 学习率

    #定义损失函数，计算均方差
    #loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)
    loss_func = torch.nn.L1Loss()      # 预测值和真实值的误差计算公式 (均方差)

    #----------------使用cuda进行GPU计算-----------------

    net.cuda()
    loss_func.cuda()

    #----------------具体训练过程-----------------

    for epoch in range(1):
        for step, (batch_x, batch_y) in enumerate(loader):

            prediction = net( batch_x.cuda() )     # input x and predict based on x

            loss = loss_func(prediction, batch_y.cuda())     # must be (1. nn output, 2. target)

            print ("loss")
            print (loss)

            #loss_plot[t,:] = loss # 将损失函数赋值给绘画变量

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            print ('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| betch y: ', batch_y.numpy())

            #计算误差百分数,并储存在data_plot当中
           
            percent = 100*(prediction - batch_y.cuda())/batch_y.cuda()

            data_plot[step,:] = percent[0,:] # 取precent矩阵的第一行

            print("data_plot")
            print(data_plot)

            print ("percent")
            print (percent)
               
            #print("\n\nprediction")
            #print(prediction)

    #计算3个输出的平均值（把每列加起来）

    #将数据从tensor转化成numpy
    data_plot_numpy = data_plot.detach().numpy()

    #print("data_plot_numpy")
    #print(data_plot_numpy[1900:2000,:])

    print("data_plot_numpy转置")
    print(data_plot_numpy.T[:,1900:2000])

    #取每个元素的绝对值
    data_plot_numpy_abs = np.abs(data_plot_numpy.T) #需要进行转置才能得到相关数据

    print("data_plot_numpy_abs")
    print(data_plot_numpy_abs[ :,1900:2000])
    print(data_plot_numpy_abs.shape)

    #调用sum函数，将每列数据加起来，求误差的平均数
    average = np.sum(data_plot_numpy_abs, axis=0)/3 #要除3，表示加起来求平均

    print("average")
    print(average[6000:6100])
    print(average.shape)

    print("average排序")
    print(np.argsort(average[6000:6100]))
   
    #将误差平均值可视化
    X = np.linspace(1, 15000, 15000, endpoint=True)

    plt.xlim(0, 7500)#设置XY轴的显示范围
    #plt.ylim(0, 50)

    plt.plot(X,average)
    plt.show()

    ##将X、Y、Z误差可视化
    #for num in range (0,7):
    #    X = np.linspace(1, times, times, endpoint=True)
    #    plt.plot(X,data_plot[:,num].detach().numpy())

    #    #plt.xlim(1, times)#设置XY轴的显示范围
    #    #plt.ylim(0, 100)

    #    plt.show()

    ##----------------将loss函数可视化-----------------


    ##将数据从tensor转化成numpy
    #loss_plot_numpy = loss_plot.detach().numpy()

    #print("data_plot_numpy")
    #print(loss_plot_numpy[1:20,:])
    #print(loss_plot_numpy.shape)

    #print("loss_plot_numpy转置")
    #print(loss_plot_numpy.T[:,1:200])
    #print(loss_plot_numpy.T.shape)

    ##取每个元素的绝对值
    #loss_plot_numpy_abs = np.abs(loss_plot_numpy.T) #需要进行转置才能得到相关数据

    ##调用sum函数，将每列数据加起来，求误差的平均数
    #average = np.sum(loss_plot_numpy_abs, axis=0)/3 #要除3，表示加起来求平均


    ##将误差平均值可视化
    #X = np.linspace(1, times, times, endpoint=True)

    ##plt.xlim(times-200, times)#设置XY轴的显示范围
    ##plt.ylim(0, 50)

    #plt.plot(X,average)
    #plt.show()


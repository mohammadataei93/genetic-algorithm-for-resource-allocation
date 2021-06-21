import numpy as np
import matplotlib.pyplot as plt  
import random    

class sensor():
    def __init__(self,name,Type,x,y,r):
        self.name=name
        self.x=x
        self.y=y
        self.r=r
        self.Type=Type
        self.parent1=None
        self.parent2=None
    def __repr__(self):
        return self.name        
        
class Area():
    def __init__(self,w,h):
        self.W=w
        self.H=h        
        
class basic_result(object):
    def __init__(self,name,sensors=[]):
        self.name=name
        self.sensors=sensors
        #self.vertical_array=[]
        self.acc=None
      
    
    def fitness_function(self):
        A_tot=0
        A_S=0
        sensors_list=self.sensors.copy()
        sensors_list2=[]
        for sen in sensors_list:
            if (sen.x >= sen.r and sen.x <= 100-sen.r and sen.y >= sen.r and sen.y <= 100-sen.r):
                sensors_list2.append(sen)
        for sen in sensors_list2:
            A_S+=(np.pi)*(sen.r**2)
        while sensors_list2:
            sens1=sensors_list2.pop()
            for sens2 in sensors_list2:
                (t1x,t1y),(t2x,t2y)=get_intercetions(sens1,sens2)
                if (t1x==None and t1y==None and t2x==None and t2y==None) :
                    continue
                if (t1x==None and t1y==None and t2x==None ) and (t2y==66 or t2y==55) :
                    A_tot+= (np.pi)*(np.minimum(sens1.r,sens2.r)**2)
                    continue
                r1=np.maximum(sens1.r,sens2.r)
                r2=np.minimum(sens1.r,sens2.r)
                T=np.sqrt((t1x-t2x)**2 + (t1y-t2y)**2)
                d1=np.sqrt(r1**2 - (T/2)**2)
                d2=np.sqrt(r2**2 - (T/2)**2)
                A_int=(r1**2 * np.arccos(d1/r1))-(d1* np.sqrt(r1**2 - d1**2))+(r2**2 * np.arccos(d2/r2))-(d2* np.sqrt(r2**2 - d2**2))
                
                A_tot+=A_int
        self.acc= (A_S - A_tot)/10000
        
    def init_res(self): 
        list_s_type1=[]       
        for i in range(5):
            list_s_type1.append(sensor(f's_type1_{i}',1,None,None,14))
        list_s_type2=[]       
        for i in range(5):
            list_s_type2.append(sensor(f's_type2_{i}',2,None,None,11.2))
        list_s_type3=[]       
        for i in range(7):
            list_s_type3.append(sensor(f's_type3_{i}',3,None,None,8.96))
        list_s=list_s_type1+list_s_type2+list_s_type3 
        self.sensors=list_s
        
        

def get_intercetions(c0,c1):
    d=np.sqrt((c1.x-c0.x)**2 + (c1.y-c0.y)**2)
    # non intersecting
    if d > c0.r + c1.r :
        return (None,None),(None,None)
    # One circle within other
    if d < abs(c0.r-c1.r):
        return (None,None),(None,66)
    # coincident circles
    if d == 0 and c0.r == c1.r:
        return  (None,None),(None,55)
    else:
        a=(c0.r**2-c1.r**2+d**2)/(2*d)
        h=np.sqrt(c0.r**2-a**2)
        x2=c0.x+a*(c1.x-c0.x)/d   
        y2=c0.y+a*(c1.y-c0.y)/d   
        x3=x2+h*(c1.y-c0.y)/d     
        y3=y2-h*(c1.x-c0.x)/d 
        x4=x2-h*(c1.y-c0.y)/d
        y4=y2+h*(c1.x-c0.x)/d
        return (x3 , y3) ,(x4 , y4)

def oclidian_dist(sensor1,sensor2):
    return np.sqrt((sensor1.x-sensor2.x)**2 + (sensor1.y-sensor2.y)**2)

          
area=Area(100,100)    
  
def init_pop(num,Try=1000):    
    basic_results=[]
    for n in range(num):
        print(f'{n*100/num} %')
        r1=basic_result(f'r{n+1}',[])  
        list_s_type1=[]       
        for i in range(5):
            list_s_type1.append(sensor(f's_type1_{i}',1,None,None,14))
            
        list_s_type2=[]       
        for i in range(5):
            list_s_type2.append(sensor(f's_type2_{i}',2,None,None,11.2))
            
        list_s_type3=[]       
        for i in range(7):
            list_s_type3.append(sensor(f's_type3_{i}',3,None,None,8.96))
                
        list_s=list_s_type1+list_s_type2+list_s_type3  
                
        random.shuffle(list_s)
        first_sensor=list_s.pop() 
        list_s=list_s_type3+list_s_type2+list_s_type1 
        list_s.remove(first_sensor)
        
        def place_first(sens):
            sens.x=np.random.choice([sens.r,(area.W-sens.r)])
            sens.y=np.random.choice([sens.r,(area.H-sens.r)])
        place_first(first_sensor)    
        
        
        def oclidian_dis(sensor1,sensor2):
            return np.sqrt((sensor1.x-sensor2.x)**2 + (sensor1.y-sensor2.y)**2)
        
        r1.sensors.append(first_sensor)
           
        while list_s:
            new_sens=list_s.pop()
            delta=0
            cant_find_position=True
            while cant_find_position:
                limit=len(r1.sensors)
                for i in range(Try):
                    new_sens.x=random.uniform(new_sens.r,area.W-new_sens.r)
                    new_sens.y=random.uniform(new_sens.r,area.H-new_sens.r)
                    check=0
                    for old_sens in r1.sensors:
                        if oclidian_dis(new_sens,old_sens)+delta >= new_sens.r+old_sens.r: 
                            if new_sens.x - new_sens.r >= -delta and new_sens.x + new_sens.r <=area.W + delta :
                                if new_sens.y-new_sens.r> -delta and new_sens.y+new_sens.r<=area.H+delta:
                                    check+=1
                    if check >=len(r1.sensors):
                        r1.sensors.append(new_sens)
                        break
                if(len(r1.sensors)!=limit):
                    break
                delta+=0.2

        basic_results.append(r1)
    return basic_results

def draw_res(res):
    circle=[]
    for sen in res.sensors:
        if sen.Type==1:
            circle.append(plt.Circle((sen.x, sen.y), sen.r, color='r'))
        if sen.Type==2:
            circle.append(plt.Circle((sen.x, sen.y), sen.r, color='g'))
        if sen.Type==3:    
            circle.append(plt.Circle((sen.x, sen.y), sen.r, color='blue'))
    fig, ax = plt.subplots()
    plt.axis([0,100,0,100])
    for cir in circle:
        ax.add_artist(cir) 

'''res=init_pop(2)
for r in res:
    draw_res(r) ''' 
#draw_res(res[1])
 

def LX_CrossOver(res1,res2):
    alpha=random.uniform(0,1)
    if alpha >= 0.5 : beta=0.5* np.log(alpha)
    else : beta= -0.5* np.log(alpha)
    sensor_list1=res1.sensors.copy()
    sensor_list2=res2.sensors.copy()
    random.shuffle(sensor_list1)
    random.shuffle(sensor_list2)
    r1=basic_result('r1',[])
    r2=basic_result('r2',[])
    Sen1T1,Sen1T2,Sen1T3=[],[],[]
    for sen in sensor_list1:
        if sen.Type==1: Sen1T1.append(sen)
        if sen.Type==2: Sen1T2.append(sen)
        if sen.Type==3: Sen1T3.append(sen)
    Sen2T1,Sen2T2,Sen2T3=[],[],[]
    for sen2 in sensor_list2:
        if sen2.Type==1: Sen2T1.append(sen2)
        if sen2.Type==2: Sen2T2.append(sen2)
        if sen2.Type==3: Sen2T3.append(sen2)   
    while Sen1T1:
        ofs_sens1=sensor('sens',1,None,None,14)
        ofs_sens2=sensor('sens',1,None,None,14)
        sen1=Sen1T1.pop()
        sen2=Sen2T1.pop()
        ofs_sens1.parent1=sen1
        ofs_sens1.parent2=sen2
        ofs_sens2.parent1=sen1
        ofs_sens2.parent2=sen2
        ofs_sens1.x=sen1.x + beta*np.abs(sen1.x - sen2.x)
        ofs_sens1.y=sen1.y + beta*np.abs(sen1.y - sen2.y)
        ofs_sens2.x=sen2.x + beta*np.abs(sen2.x - sen1.x)
        ofs_sens2.y=sen2.y + beta*np.abs(sen2.y - sen1.y)
        r1.sensors.append(ofs_sens1)
        r2.sensors.append(ofs_sens2)
    while Sen1T2:
        ofs_sens1=sensor('sens',2,None,None,11.2)
        ofs_sens2=sensor('sens',2,None,None,11.2)
        sen1=Sen1T2.pop()
        sen2=Sen2T2.pop()
        ofs_sens1.parent1=sen1
        ofs_sens1.parent2=sen2
        ofs_sens2.parent1=sen1
        ofs_sens2.parent2=sen2
        ofs_sens1.x=sen1.x + beta*np.abs(sen1.x - sen2.x)
        ofs_sens1.y=sen1.y + beta*np.abs(sen1.y - sen2.y)
        ofs_sens2.x=sen2.x + beta*np.abs(sen2.x - sen1.x)
        ofs_sens2.y=sen2.y + beta*np.abs(sen2.y - sen1.y)
        r1.sensors.append(ofs_sens1)
        r2.sensors.append(ofs_sens2)
    while Sen1T3:
        ofs_sens1=sensor('sens',3,None,None,8.96)
        ofs_sens2=sensor('sens',3,None,None,8.96)
        sen1=Sen1T3.pop()
        sen2=Sen2T3.pop()
        ofs_sens1.parent1=sen1
        ofs_sens1.parent2=sen2
        ofs_sens2.parent1=sen1
        ofs_sens2.parent2=sen2
        ofs_sens1.x=sen1.x + beta*np.abs(sen1.x - sen2.x)
        ofs_sens1.y=sen1.y + beta*np.abs(sen1.y - sen2.y)
        ofs_sens2.x=sen2.x + beta*np.abs(sen2.x - sen1.x)
        ofs_sens2.y=sen2.y + beta*np.abs(sen2.y - sen1.y)
        r1.sensors.append(ofs_sens1)
        r2.sensors.append(ofs_sens2)

    return r1,r2
        
def AMXO_CrossOver(res1,res2):
    alpha=random.uniform(0,1)
    sensor_list1=res1.sensors.copy()
    sensor_list2=res2.sensors.copy()
    random.shuffle(sensor_list1)
    random.shuffle(sensor_list2)
    r1=basic_result('r1',[])
    r2=basic_result('r2',[])
    Sen1T1,Sen1T2,Sen1T3=[],[],[]
    for sen in sensor_list1:
        if sen.Type==1: Sen1T1.append(sen)
        if sen.Type==2: Sen1T2.append(sen)
        if sen.Type==3: Sen1T3.append(sen)
    Sen2T1,Sen2T2,Sen2T3=[],[],[]
    for sen2 in sensor_list2:
        if sen2.Type==1: Sen2T1.append(sen2)
        if sen2.Type==2: Sen2T2.append(sen2)
        if sen2.Type==3: Sen2T3.append(sen2)   
    while Sen1T1:
        ofs_sens1=sensor('sens',1,None,None,14)
        ofs_sens2=sensor('sens',1,None,None,14)
        sen1=Sen1T1.pop()
        sen2=Sen2T1.pop()
        ofs_sens1.parent1=sen1
        ofs_sens1.parent2=sen2
        ofs_sens2.parent1=sen1
        ofs_sens2.parent2=sen2
        ofs_sens1.x=(alpha*sen1.x)+((1-alpha)*sen2.x)
        ofs_sens1.y=(alpha*sen1.y)+((1-alpha)*sen2.y)
        ofs_sens2.x=(alpha*sen2.x)+((1-alpha)*sen1.x)
        ofs_sens2.y=(alpha*sen2.y)+((1-alpha)*sen1.y)
        r1.sensors.append(ofs_sens1)
        r2.sensors.append(ofs_sens2)
    while Sen1T2:
        ofs_sens1=sensor('sens',2,None,None,11.2)
        ofs_sens2=sensor('sens',2,None,None,11.2)
        sen1=Sen1T2.pop()
        sen2=Sen2T2.pop()
        ofs_sens1.parent1=sen1
        ofs_sens1.parent2=sen2
        ofs_sens2.parent1=sen1
        ofs_sens2.parent2=sen2
        ofs_sens1.x=(alpha*sen1.x)+((1-alpha)*sen2.x)
        ofs_sens1.y=(alpha*sen1.y)+((1-alpha)*sen2.y)
        ofs_sens2.x=(alpha*sen2.x)+((1-alpha)*sen1.x)
        ofs_sens2.y=(alpha*sen2.y)+((1-alpha)*sen1.y)
        r1.sensors.append(ofs_sens1)
        r2.sensors.append(ofs_sens2)
    while Sen1T3:
        ofs_sens1=sensor('sens',3,None,None,8.96)
        ofs_sens2=sensor('sens',3,None,None,8.96)
        sen1=Sen1T3.pop()
        sen2=Sen2T3.pop()
        ofs_sens1.parent1=sen1
        ofs_sens1.parent2=sen2
        ofs_sens2.parent1=sen1
        ofs_sens2.parent2=sen2
        ofs_sens1.x=(alpha*sen1.x)+((1-alpha)*sen2.x)
        ofs_sens1.y=(alpha*sen1.y)+((1-alpha)*sen2.y)
        ofs_sens2.x=(alpha*sen2.x)+((1-alpha)*sen1.x)
        ofs_sens2.y=(alpha*sen2.y)+((1-alpha)*sen1.y)
        r1.sensors.append(ofs_sens1)
        r2.sensors.append(ofs_sens2)
        
    return r1,r2


def Mutation(res):
    random.shuffle(res.sensors)
    mut_sens=res.sensors.pop()
    mut_sens.x=mut_sens.x + np.random.normal(0,(mut_sens.parent1.x - mut_sens.parent2.x)**2)
    mut_sens.y=mut_sens.y + np.random.normal(0,(mut_sens.parent1.y - mut_sens.parent2.y)**2)
    res.sensors.append(mut_sens)
    return res

def VFA(res,Try,step):
    for sen in res.sensors:
        for i in range(Try):
            last_acc=res.acc
            randx=np.random.uniform(-step,step)
            randy=np.random.uniform(-step,step)
            sen.x+=randx
            sen.y+=randy
            res.fitness_function()
            if res.acc <= last_acc:
                sen.x-=randx
                sen.y-=randy
    return res

def update_pop(pop):
    acc_list1=[]
    for r in pop:
        r.fitness_function()
        acc_list1.append(r.acc)
    return acc_list1



def GENETIC_ALGHORITHM(init_pop_num=50,LXperAMXO=0.1,mutation=True,epochs=150,VFA_Try=500,VFA_step=0.05,draw_av=True):
    first_pop=init_pop(init_pop_num)
    population=first_pop.copy()
    for i in range(epochs):
        print(f'{i*100/epochs} %')
        random.shuffle(population)
        parents=population.copy()
        acc_list=update_pop(population)    
        while parents:
            par1=parents.pop()
            par2=parents.pop()
            CrossOver_Type=np.random.uniform(0,1)
            if CrossOver_Type < LXperAMXO:
                ofs1,ofs2 = LX_CrossOver(par1,par2)
                if mutation:
                    ofs1=Mutation(ofs1)
                    ofs2=Mutation(ofs2)
            if CrossOver_Type >= LXperAMXO:
                ofs1,ofs2 = AMXO_CrossOver(par1,par2) 
                if mutation:
                    ofs1=Mutation(ofs1)
                    ofs2=Mutation(ofs2)
            ofs1.fitness_function()
            ofs2.fitness_function()
            if ofs1.acc >= min(acc_list):
                for r in population:
                    if r.acc==min(acc_list):
                        population.remove(r)
                        population.append(ofs1)
                        acc_list=update_pop(population)
            if ofs2.acc >= min(acc_list):
                for r in population:
                    if r.acc==min(acc_list):
                        population.remove(r)
                        population.append(ofs2)            
                        acc_list=update_pop(population)
    for p in population:
        VFA(p,VFA_Try,VFA_step)
        if draw_av: draw_res(p)
    acc_list=update_pop(population)
    return population,acc_list




pop,acc = GENETIC_ALGHORITHM(mutation=False)



    

    























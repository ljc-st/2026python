class Sort:#快排 时间：nlogn 空间:logn 最坏:n^2
    def __init__(self,n):
        self.len=n
        self.arr=[3,87,2,93,78,56,61,38,12,40]
    #def random_data(self):
        #for i in range(self.len):
    def partition(self,left,right):
        arr=self.arr
        k=i=left
        for i in range(left,right):
            if arr[i]<arr[right]:
                arr[i],arr[k]=arr[k],arr[i]
                k+=1
        arr[k],arr[right]=arr[right],arr[k]
        return k
    def quick_sort(self,left,right):
        if left < right: #注意不是while哦
            pivot=self.partition(left,right)
            self.quick_sort(left,pivot-1)
            self.quick_sort(pivot+1,right)
    def adjust_max_heap(self,pos,arr_len):
        arr=self.arr
        dad=pos
        son=2*dad+1
        while son < arr_len: #超出边界
            if son+1<arr_len and arr[son]<arr[son+1]:#右孩子存在 且 右孩子大于左孩子
                son+=1
            if arr[son]>arr[dad]:
                arr[son],arr[dad]=arr[dad],arr[son]
                dad=son
                son=2*dad+1
            else:
                break

    def heap_sort(self):
        # 把列表调整为大根堆
        for parent in range(self.len//2-1,-1,-1):#从后往前遍历
            self.adjust_max_heap(parent,self.len)
        arr=self.arr
        arr[0],arr[self.len-1]=arr[self.len-1],arr[0]#堆顶元素和最后一个元素交换
        for arr_len in range(self.len-1,1,-1):
            self.adjust_max_heap(0,arr_len)
            arr[0], arr[arr_len - 1] = arr[arr_len - 1], arr[0]


if __name__=='__main__':
    my_sort=Sort(10)
    print(my_sort.arr)
    my_sort.quick_sort(0,9)
    print(my_sort.arr)
    my_sort.heap_sort()
    print(my_sort.arr)
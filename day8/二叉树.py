class Node:
    def __init__(self,elem=-1,lchild=None,rchild=None):
        self.elem=elem
        self.lchild=lchild
        self.rchild=rchild
class BinaryTree:
    def __init__(self):
        self.root=None
        self.help_queue=[]#辅助队列
    def level_build_tree(self,node:Node):
        if self.root is None:#根为空
            self.root=node
            self.help_queue.append(node)#入队
        else:
            self.help_queue.append(node)
            if self.help_queue[0].lchild is None:
                self.help_queue[0].lchild=node
            else:
                self.help_queue[0].rchild = node
                self.help_queue.pop(0)#父节点出队
    def pre_order(self,current_node:Node):
        if current_node:
            print(current_node.elem,end=' ')
            self.pre_order(current_node.lchild) #递归调用
            self.pre_order(current_node.rchild)
    def mid_order(self,current_node:Node):
        if current_node:
            self.mid_order(current_node.lchild) #递归调用
            print(current_node.elem,end=' ')
            self.mid_order(current_node.rchild)
    def last_order(self,current_node:Node):
        if current_node:

            self.last_order(current_node.lchild)
            self.last_order(current_node.rchild)
            print(current_node.elem,end=' ')
             #递归调用
    def level_order(self):
        help_queue=[]
        help_queue.append(self.root)#根节点入队
        while help_queue:
            out_node:Node=help_queue.pop(0)
            print(out_node.elem,end=' ')
            if out_node.lchild:
                help_queue.append(out_node.lchild)
            if out_node.rchild:
                help_queue.append(out_node.rchild)


if __name__=='__main__':
    tree=BinaryTree()
    for i in range(1,11):
        new_node=Node(i) #实例化节点
        tree.level_build_tree(new_node) #把节点放入树中
    tree.pre_order(tree.root) #根左右 前序遍历 也是深度优先遍历
    print()
    tree.mid_order(tree.root)
    print()
    tree.last_order(tree.root)
    print()
    tree.level_order()
    print()
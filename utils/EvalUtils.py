import numpy as np

class Evaluation:
    def __init__(self,num_classes=10,calc_class=1):
        '''
        Args:
        :param classnum: 类别数量
        :param calc_class: 控制计算哪一个类f1和acc，值为-1的时候计算平均值
        '''
        self.num_classes = num_classes
        self.calc_class=calc_class

    def precision(self,TP,FP):
        return TP/(TP+FP)

    def recall(self,TP,FN):
        return TP/(TP+FN)

    def accuracy(self,PreRight_num,total_num):
        return PreRight_num/total_num

    def IoU(self,TP,FP,FN):
        return TP/(TP+FP+FN)

    def ConfusionMatrix(self,pred, groundtruth):
        '''
        混淆矩阵：CM[I,J]表示i预测成j。
        '''
        CM = np.zeros((self.num_classes, self.num_classes))
        for clai in range(self.num_classes):
            mask = (groundtruth == clai)
            for claj in range(self.num_classes):
                CM[clai, claj] = np.sum(pred[mask] == claj)
        return CM

    def FWIoU(self,freq,iou):
        return freq*iou

    def all_(self,output,mask):
        '''
        :param output:预测结果
        :param mask: gt
        :return:
        '''
        output=np.array(output).reshape(-1)
        mask=np.array(mask).reshape(-1)
        nonan=(mask>=0)
        CM = self.ConfusionMatrix(output[nonan],mask[nonan])
        # p1 预测为i的个数
        p1 = np.sum(CM, axis=0)
        # p2 类别i的所有预测和，即类别i的个数
        p2 = np.sum(CM, axis=1)
        ious = []
        fwious = []
        pres = []
        recs = []
        f1s = []
        totalnum = np.sum(p2)
        PreRight=0
        # 排除某些类别样本个数是0
        num_classes = np.sum(p2 != 0)
        for cla in range(self.num_classes):
            TP = CM[cla][cla]
            FN = p2[cla] - TP
            FP = p1[cla] - TP
            PreRight+=TP
            iou = self.IoU(TP, FN, FP)
            print(iou)
            ious.append(iou)
            # 类别cla出现的频率 = 类别cla的个数/总个数
            freq = p2[cla]/totalnum
            fwious.append(self.FWIoU(freq,iou))
            pres.append(self.precision(TP, FP))
            recs.append(self.recall(TP,FN))
            f1s.append(self.IoU(TP, FN, FP))

        # acc是一种衡量全局的指标（预测正确的/样本总个数）
        acc = self.accuracy(PreRight,totalnum)
        miou = np.nansum(ious) / num_classes
        fwiou = np.nansum(fwious)
        if self.calc_class==-1:
            mpre = np.nansum(pres) / (num_classes)
            mrec = np.nansum(recs) / (num_classes)
            if mpre + mrec == 0:
                return "无效"
            else:
                mF1=format(2 * mpre * mrec / (mpre + mrec), '.3%')
                return [mF1, format(acc, '.3%'), format(miou, ".3%"),format(fwiou, ".3%")]
        else:
            pre = pres[self.calc_class]
            rec = recs[self.calc_class]
            if pre+rec==0:
                return "无效"
            else:
                F1=format(2 * pre * rec / (pre + rec), '.3%')
                return [F1,format(acc,'.3%'),format(miou,".3%"),format(fwiou, ".3%")]

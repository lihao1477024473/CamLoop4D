
import torch
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# name = "bear_maskAllOneCls_py"
# # path_ckpt = fr"D:\AIGC\code\shape-of-motion-main\lh\{name}\checkpoints\last.ckpt"
# path_ckpt = fr"D:\AIGC\code\som-change\lh\lily-dragon_lemniscate_maskAllOneCls_py_change_sample05_mofBase\checkpoints\last.ckpt"

dataName="backpack" # paper-windmill,spin,backpack
out="lh30"
path_ckpt =  f"/root/autodl-tmp/som-change/{out}/{dataName}/checkpoints/last.ckpt"


checkpoint = torch.load(path_ckpt,weights_only=False)
print(checkpoint.keys())  # 查看包含哪些键
#
# means = checkpoint['model']['fg.params.means'].cpu().numpy()
# colors = checkpoint['model']['fg.params.colors'].cpu().numpy()
# colors = (colors - colors.min()) / (colors.max() - colors.min())  # 线性归一化
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(means[:,0], means[:,1], means[:,2], c=colors, s=0.1)
# plt.show()
# exit()



for k,v in checkpoint.items():
    print(k,type(v))
    if k == "model":
        for k1,v1 in v.items():
            if k1 in ["motion_bases.params.rots",
                    #   "motion_bases.params.rots_fronze",
                      "motion_bases.rots_frozen",
                      "motion_bases.params.transls",
                    #   "motion_bases.params.transls_fronze",
                      "motion_bases.transls_frozen",
                      "fg.params.motion_coefs"]:
                print(k1,v1.shape,v1)
            else:
                print(k1,v1.shape)
    if k == "optimizers":
        print(k,v.keys())
        for k_opt,v_opt in v.items():
            print(f"     ",k_opt)
            # for k_opt1,v_opt1 in v_opt.items():
            #     print(f"         {k_opt1}:",k_opt1,type(v_opt1))




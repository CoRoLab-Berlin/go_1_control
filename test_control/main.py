# add path
import sys

import mujoco
import mujoco_viewer
import numpy as np
import qmt

sys.path.append('./jax_files')
import matplotlib.pyplot as plt
from com_pos_jax import com_pos
from dynamics.gravity_vetor_jax import gravity_vetor
from J_pcom_jax import J_pcom
from kinematics.J_pos_fk_trunk_FL_foot_jax import J_pos_fk_trunk_FL_foot
from kinematics.J_pos_fk_trunk_FR_foot_jax import J_pos_fk_trunk_FR_foot
from kinematics.J_pos_fk_trunk_RL_foot_jax import J_pos_fk_trunk_RL_foot
from kinematics.J_pos_fk_trunk_RR_foot_jax import J_pos_fk_trunk_RR_foot


def main():
    model = mujoco.MjModel.from_xml_path('./xml/go1.xml')
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)

    # q_0 =  np.array([0.0055,  
    #    0.011,   
    #    0.058,   
    #    1,       
    #   -9.2e-06, 
    #    1.2e-05, 
    #   -0.027,   
    #   -0.54,    
    #    1.1,     
    #   -2.7,     
    #    0.56,    
    #    1.1,     
    #   -2.7,     
    #   -0.56,    
    #    1.1,     
    #   -2.7,     
    #    0.54,    
    #    1.1,     
    #   -2.7, 
    # ])
    q_0 =  np.array([
    0,       
    0,       
    0.3,    
    1,       
    0,       
    0,       
    0,       
    -0.1,       
    0.81,    
    -1.8,     
    0.1,       
    0.81,    
    -1.8,     
    -0.1,       
    0.81,    
    -1.8,     
    0.1,       
    0.81,    
    -1.8, 
    ])

    data.qpos =  q_0
    mujoco.mj_forward(model, data)

    def get_qpos(data):
        q_pos = np.zeros(18)
        q_pos[:3] = data.qpos[:3]
        q_pos[3:6] = qmt.eulerAngles(data.qpos[3:7], 'zyx')
        q_pos[6:] = data.qpos[7:]
        return q_pos


    mass = 11.309932 - 4 * (0.218015)
    force = np.array([0, 0, -mass*9.81])
    wrench = np.zeros(6)
    wrench[:3] = force
    front = 0.448 # 0.5 # 0.445
    back = 1 - front


    total_cmds = []
    pos_com = []
    for i in range(500):
        qpos = get_qpos(data)

        pos_com.append(com_pos(qpos))
        
        ctrl = np.zeros(12)
        ctrl[0:3]  = np.array((J_pos_fk_trunk_FR_foot(qpos)).T@(force * front/2))[6:9]
        ctrl[3:6] += np.array((J_pos_fk_trunk_FL_foot(qpos)).T@(force * front/2))[9:12]
        ctrl[6:9] += np.array((J_pos_fk_trunk_RR_foot(qpos)).T@(force * back/2))[12:15]
        ctrl[9:]  += np.array((J_pos_fk_trunk_RL_foot(qpos)).T@(force * back/2))[15:18]
        ctrl = np.clip(ctrl, -30, 30)
        data.ctrl = ctrl
        total_cmds.append(ctrl)
        
        mujoco.mj_step(model, data)
        
        viewer.render()
    viewer.close()

    total_ctrls = np.array(total_cmds)

    fig = plt.figure()
    ax_dict = fig.subplot_mosaic(
        [
            ['hip'],
            ['thigh'],
            ['calf'],

        ]
    )

    ax_dict['hip'].plot(total_ctrls[:,0::3])
    ax_dict['hip'].legend(['FR', 'FL', 'RR', 'RL'])
    ax_dict['hip'].grid()

    ax_dict['thigh'].plot(total_ctrls[:,1::3])
    ax_dict['thigh'].legend(['FR', 'FL', 'RR', 'RL'])
    ax_dict['thigh'].grid()

    ax_dict['calf'].plot(total_ctrls[:,2::3])
    ax_dict['calf'].legend(['FR', 'FL', 'RR', 'RL'])
    ax_dict['calf'].grid()

    fig.savefig('cmds.png')
    plt.show()

    plt.plot(np.array(pos_com), label=['com_x', 'com_y', 'com_z'])
    plt.legend()
    plt.grid()
    plt.savefig('com.png')
    plt.show()


if __name__ == '__main__':
    main()
    main()

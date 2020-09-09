case_generato.pyr:生成新的案例<br/>
grid_definer.py:定义电网拓扑及参数<br/>
dql_learn.py:强化学习训练<br/>
dqn_model.py:神经网络<br/>
main.py:主程序<br/>
<br/><br/>
utils<br/>
--env.py:封装潮流环境<br/>
--pp_wrapper.py:pandapower API<br/>
--replay_buffer.py:loader，提供训练样本<br/>
--schedule.py:在exploitation和exploration之间权衡<br/>
<br/><br/>
config<br/>
--expr.yaml:实验配置<br/>
--gen_case.yaml:案例生成配置<br/>
--grid.yaml:网络生成配置<br/>
% WPV1_eco_analyse_demo - V7 (add variable control)
% =================================================================
% 修正了电解槽物理模型中的关键错误(参考文献大错误)，修正发电机组约束定义错误(隐式约束)，完善数据结果可视化部分
clear;
%% 输入条件
T=24;  % 24 or 24*365
isLargeWindSea = 1;
flag = 1;   % 1风光互补  2纯风 3纯光

target_NH3_sale = 10^8/365;  % 目标氨产量，固定用于分析风光互补的优势。 2800 3000 3100

% 测试用数据，风峰不对
% 后续数据可xlsread()读取excel表格
T_env=25*ones(25,1);
G_C = [0,0,0,0,100,200,400,600,800,900,1000,950,900,800,600,400,200,100,0,0,0,0,0,0,0].'+242;
v = [8,8,9,9,10,11,12,12.5,13,12,11,10,10,9,9,8,8,7,7,8,8,9,9,8,8].';

%% 决策变量、约束条件
con = [];
% 发电机组  PV WP
E_Gen = sdpvar(2,1,'full'); E_PV = E_Gen(1); E_WP = E_Gen(2);
con = [con, E_Gen >= 0];
% 化工生产单元(cp) ele N2 NH3
R_ate_cp = sdpvar(3,1,'full'); R_ate_ele_cp = R_ate_cp(1); R_ate_N2_cp = R_ate_cp(2); R_ate_NH3_cp = R_ate_cp(3);
con = [con, R_ate_cp >= 0];
% 燃料电池单元(pg) H2 NH3
R_ate_pg = sdpvar(2,1,'full'); R_ate_H2_pg = R_ate_pg(1); R_ate_NH3_pg = R_ate_pg(2);
con = [con, R_ate_pg >= 0];
% 储能单元(st) H2 NH3
R_ate_st = sdpvar(2,1,'full'); R_ate_H2_st = R_ate_st(1); R_ate_NH3_st = R_ate_st(2);
con = [con, R_ate_st >= 0];

% 为决策变量设置合理上界，防止无界
con = [con, E_Gen <= 5000000000];     % kW
con = [con, R_ate_cp <= 20000000000];  % kg/h
con = [con, R_ate_pg <= 200000000000];
con = [con, R_ate_st <= 5000000000000];  % t


% =====隐式约束在Yalmip中会出大问题！  采用显式约束！
%{
% 光伏模块(PV)
G_N = 1000; T_N = 25; gamma = -0.0047;
T_C = T_env + 30/1000*G_C;
P_PV = E_PV*G_C./G_N.*(1+gamma.*(T_C-T_N));
P_PV(G_C >= G_N) = E_PV;
% 风电模块(WP)
v_ci = 3; v_R = 12.5; v_co = 25;
P_WP = (v.^3-v_ci.^3)./(v_R.^3-v_ci.^3)*E_WP;
P_WP(v<=v_ci | v>=v_co) = 0;
P_WP(v>=v_R & v<v_co) = E_WP;
%}
% 将P_PV和P_WP定义为完整的sdpvar向量
P_PV = sdpvar(T+1,1,'full');
P_WP = sdpvar(T+1,1,'full');

% --- 光伏模块(PV) - 使用显式约束 ---
G_N = 1000; T_N = 25; gamma = -0.0047;
T_C = T_env + 30/1000*G_C;
P_PV_pot = E_PV*G_C./G_N.*(1+gamma.*(T_C-T_N)); % 计算未满发电量

con = [con, P_PV >= 0]; % 发电量不能为负
% con = [con, P_PV(t) <= E_PV];
for t = 1:T+1
    if G_C(t) >= G_N
        con = [con, P_PV(t) == E_PV];
    else
        con = [con, P_PV_pot(t) == P_PV(t)];
    end
end

% 风电模块(WP) - 使用显式约束
v_ci = 3; v_R = 12.5; v_co = 25;
con = [con, P_WP >= 0]; % 发电量不能为负
for t = 1:T+1
    if v(t) <= v_ci || v(t) >= v_co
        % 在切入风速以下或切出风速以上，出力为0
        con = [con, P_WP(t) == 0];
    elseif v(t) > v_ci && v(t) < v_R
        % 在切入和额定风速之间，出力与风速的函数相关
        P_WP_potential = (v(t)^3 - v_ci^3) / (v_R^3 - v_ci^3) * E_WP;
        con = [con, P_WP(t) == P_WP_potential];
    elseif v(t) >= v_R && v(t) < v_co
        % 在额定风速和切出风速之间，出力为额定功率
        con = [con, P_WP(t) == E_WP];
    end
end


% 化工单元运行逻辑 (带启停控制)
b_ele = binvar(T+1,1,'full'); 
b_nh3 = binvar(T+1,1,'full');

% 电解槽
rho_H2_ele = 50; % 单位: kWh/kg (生产1kg氢气耗50度电)
rho_H2_fuel = 0.05; % 单位: kg/kWh (消耗1kg氢气发0.05度电) - 这是比热值倒数
eta_P_min_ele = 0.05;
P_H2_in = sdpvar(T+1,1);

% ==================== 物理模型修正 ====================
m_H2_in = P_H2_in / rho_H2_ele; % 产氢速率 = 功率 / 单位耗电量
% =====================================================

con = [con, P_H2_in <= R_ate_ele_cp .* b_ele];
con = [con, P_H2_in >= eta_P_min_ele * R_ate_ele_cp .* b_ele];

P_H2_out = sdpvar(T+1,1);
m_H2_fuel = P_H2_out./rho_H2_fuel; 
con = [con, 0 <= P_H2_out <= R_ate_H2_pg];

% 储氢罐 (使用硬约束，软约束没那个必要)
eta_H2_in = 0.95; eta_H2_out = 0.95;
L_H2 = sdpvar(T+1,1);
m_H2_NH3 = sdpvar(T+1,1,'full');
for t = 0:T-1
    con = [con, L_H2(t+2) == L_H2(t+1) + m_H2_in(t+1).*eta_H2_in - (m_H2_fuel(t+1) + m_H2_NH3(t+1))./eta_H2_out];
end
con = [con, 0.05.*R_ate_H2_st <= L_H2 <= R_ate_H2_st, L_H2(1)==0.05.*R_ate_H2_st, L_H2(T+1)==0.05.*R_ate_H2_st];

% 空分制氮
rho_N2 = 0.8;
P_N2 = sdpvar(T+1,1,'full');
m_N2 = rho_N2*P_N2;
con = [con, P_N2 <= R_ate_N2_cp .* b_nh3]; 
con = [con, P_N2 >= 0];

% H-B合成氨
rho_NH3 = 3.1; rho_NH3_fuel = 1.6; eta_NH3_mk = 0.98; eta_P_min_NH3 = 0.3; 
eta_delta_P_NH3_in = 0.08;
P_NH3_in = sdpvar(T+1,1,'full');
m_NH3_in = rho_NH3 * P_NH3_in;
con = [con, P_NH3_in <= R_ate_NH3_cp .* b_nh3];
con = [con, P_NH3_in >= eta_P_min_NH3 * R_ate_NH3_cp .* b_nh3];

P_NH3_out = sdpvar(T+1,1,'full');
m_NH3_fuel = P_NH3_out./rho_NH3_fuel;
con = [con, 0 <= P_NH3_out <= R_ate_NH3_pg];
% 爬坡
for t = 1:T
    con = [con, -eta_delta_P_NH3_in.*R_ate_NH3_cp <= (P_NH3_in(t+1)-P_NH3_in(t)) <= eta_delta_P_NH3_in.*R_ate_NH3_cp];
end
con = [con, P_NH3_in(1) == 0.5.*R_ate_NH3_cp .* b_nh3(1)]; 
con = [con, eta_NH3_mk.*m_N2 == 14/17.*m_NH3_in, eta_NH3_mk.*m_H2_NH3 == 3/17.*m_NH3_in];

% 储氨罐
eta_NH3_in = 0.95; eta_NH3_out = 0.95;
L_NH3 = sdpvar(T+1,1,'full');
m_NH3 = sdpvar(T+1,1,'full');
for t = 0:T-1
    con = [con, L_NH3(t+2) == L_NH3(t+1) + m_NH3_in(t+1).*eta_NH3_in - (m_NH3_fuel(t+1) + m_NH3(t+1))./eta_NH3_out];
end
con = [con, 0 <= L_NH3 <= R_ate_NH3_st, L_NH3(1)==0, L_NH3(T+1)==0];
con = [con, m_NH3 >= 0];

% 功率平衡约束
P_loss = sdpvar(T+1,1); P_buy = sdpvar(T+1,1); P_sell = sdpvar(T+1,1);
con = [con, P_WP + P_PV + P_H2_out + P_NH3_out + P_buy == P_loss + P_H2_in + P_N2 + P_NH3_in + P_sell];
con = [con, P_loss >= 0, P_buy >= 0, P_sell >= 0]; % 三者均大于0

% 弃电率与电网交互约束
epsilon = 0.1; % 弃电率约束系数
beta = 0.2; % 上网约束系数
theta = 0.05; % 净上网约束系数
beta_buy = 0.1;
con = [con, sum(P_loss) <= epsilon*sum(P_PV + P_WP)];
con = [con, sum(P_sell) <= beta.*sum(P_PV + P_WP)];
con = [con, sum(P_sell - P_buy) <= theta.*sum(P_PV + P_WP)];
% 购电约束，使火电使用率降低，产生更加绿色的绿氨
con = [con, sum(P_buy) <= beta_buy*sum(P_PV + P_WP)];

con = [con, P_sell <= 0.5*(P_PV + P_WP)];

% 风光互补出力调整 flag   1风光互补  2纯风 3纯光
if flag == 2
    con = [con, E_PV == 0]; 
elseif flag == 3
    con = [con, E_WP == 0]; 
end

% 使用严格的售氨量约束进行测试
con = [con, sum(m_NH3) == target_NH3_sale]; 

% 强制设备最小产能以满足售氨要求，其实这段可以不加，但以防万一
min_NH3_prod_total = target_NH3_sale / (eta_NH3_in * eta_NH3_out);
con = [con, sum(m_NH3_in) >= min_NH3_prod_total];
min_NH3_rate = min_NH3_prod_total / T;
con = [con, R_ate_NH3_cp >= min_NH3_rate]; 
H2_per_NH3 = (3/17) / eta_NH3_mk;
% ==================== 最小产能约束修正  防止爆零
min_H2_rate = min_NH3_rate * H2_per_NH3;
min_power_demand = min_H2_rate * rho_H2_ele; % 修正后的最小电力需求
con = [con, R_ate_ele_cp >= min_power_demand];
if flag == 1, con = [con, E_PV + E_WP >= min_power_demand];
elseif flag == 2, con = [con, E_WP >= min_power_demand];
else, con = [con, E_PV >= min_power_demand]; end
% ====================

%% 目标函数 (线性)
c_sa = 4;
E_NH3 = c_sa * sum(m_NH3);
r=0.05; L_sf=20;
alpha = r*(1+r)^L_sf/( (24*365/T) * ((1+r)^L_sf-1) );
% Gen cost
k_inv = [3200;3477]; w = [0.012;0.02];
C_GE = sum(k_inv.*E_Gen.*(alpha + w*T/(24*365))); % 关键点：同alpha，将投资和维护成本全部都变成日或者年的，上面的alpha已经优化掉了，这里很关键
c_cp_inv = [834.2; 28.35; 344.9]; c_cp_op = [3; 0.72; 0.11];
C_cp = sum(c_cp_inv.*R_ate_cp.*alpha) + sum(c_cp_op.*[sum(m_H2_in); sum(m_N2); sum(m_NH3_in)]);
c_pg_inv = [2210; 4000]; c_pg_op = [0.41; 0.31];
C_pg = sum(c_pg_inv.*R_ate_pg.*alpha) + sum(c_pg_op.*[sum(P_H2_out); sum(P_NH3_out)]);
c_st_inv = [5743.9; 0.57];
C_ST = sum(c_st_inv.*R_ate_st.*alpha);
c_se = 0.1;
E_sell = c_se*sum(P_sell);

% 购电成本 峰谷、季节电价
% 购电，峰谷电价、季节电价模型 t+1
% 大风季
c_big = zeros(24,1);
c_big(18:21) = 0.68;
c_big([5:10,16:17,22:24]) = 0.52;
c_big([1:4,11:15]) = 0.34;
% 小风季
c_sma = zeros(24,1);
c_sma(19:20) = 0.77;
c_sma([6:7,18,21]) = 0.68;
c_sma([8:10,16:17,22:24,1:5]) = 0.52;
c_sma(11:15) = 0.34;

c_pe = [];
if T == 24
    if isLargeWindSea == 1
        % 大风季 t+1
        c_pe = c_big;
    else
        % 小风季 t+1
        c_pe = c_sma;
    end
    c_pe(T+1) = c_pe(1);
elseif T == 365*24
    % 1-5月(151天)大风季，6-8月(92)小风季，9-12月(122)大风季
    c_pe = repmat(c_big,151,1);
    c_pe = [c_pe; repmat(c_sma,92,1)];
    c_pe = [c_pe; repmat(c_big,122,1)];

    c_pe(T+1) = c_pe(1);
else
    disp('T值非法');
    return;
end
C_buy = sum(c_pe.*P_buy);
% 惩罚成本，暂不考虑
C_loss = 0;
% 投资和维护成本
C_IO = C_GE + C_cp + C_pg + C_ST;
E = E_NH3 + E_sell - C_IO - C_loss - C_buy;
Objective = E;
%% 优化配置
% debug显示
ops = sdpsettings;
ops.solver = 'gurobi';
ops.verbose = 2;
ops.showprogress = 1;
% 利益最大值，取负
sol = optimize(con, -Objective, ops);

%% 结果分析
if sol.problem ~= 0
    disp("求解失败! 原因:");
    yalmiperror(sol.problem)
    
    % 检查约束可行性
    [~, ~, ~, index] = check(con);
    if any(index)
        disp("冲突约束索引: ");
        find(index)
    end
    
    % 尝试放宽约束
    disp("尝试放宽售氨约束...");
    con_relaxed = con;
    con_relaxed(end) = []; % 移除售氨约束
    sol_relaxed = optimize(con_relaxed, -Objective, ops);
    if sol_relaxed.problem == 0
        disp("放宽约束后求解成功，检查设备容量:");
        disp("光伏: "+value(E_PV)+" kW");
        disp("风电: "+value(E_WP)+" kW");
        disp("电解槽: "+value(R_ate_ele_cp)+" kg/h");
    end
    return;
end

%{
%% 结果分析
if sol.problem ~= 0
    disp("出错了!");
    sol.info
    yalmiperror(sol.problem)
    return;
end
%}

disp("求解成功");
% solution = value(x);
% 获取解的值(覆盖)

E_Gen_val = value(E_Gen);
disp("光伏装机量(kW):"+E_Gen_val(1))
disp("风电装机量(kW):"+E_Gen_val(2))

R_ate_cp_val = value(R_ate_cp);
R_ate_pg_val = value(R_ate_pg);
R_ate_st_val = value(R_ate_st);
disp("电解槽装机量(kg/h):"+R_ate_cp_val(1))
disp("空分制氮装机量(kg/h):"+R_ate_cp_val(2))
disp("H-B制氨装机量(kg/h):"+R_ate_cp_val(3))

disp("氢燃料电池装机量(kWh):"+R_ate_pg_val(1))
disp("氨燃料电池装机量(kWh):"+R_ate_pg_val(2))

disp("储氢罐容量(kg):"+R_ate_st_val(1))
disp("储氨罐容量(kg):"+R_ate_st_val(2))

% 成本展示(元)
%{
disp("光伏投资成本:"+k_inv(1).*E_Gen_val(1).*alpha);
disp("光伏维护成本:"+k_inv(1).*E_Gen_val(1).*alpha);
disp("风电投资成本:"+k_inv(2).*E_Gen_val(2).*alpha);
disp("风电维护成本:"+k_inv(2).*E_Gen_val(2).*alpha);
%}
disp("发电模块总成本:"+value(C_GE))
disp("化工生产单元模块总成本:"+value(C_cp))
disp("燃料电池模块总成本:"+value(C_pg))
disp("存储单元模块总成本:"+value(C_ST))

disp("购电成本:"+value(C_buy))
disp("售电收益:"+value(E_sell))
disp("弃电惩罚成本:"+value(C_loss))
disp("弃电量(kWh):"+value(sum(P_loss)))

disp("弃电率:"+value( sum(P_loss) / (sum(P_PV + P_WP)) )*100+"%" )
disp("售电率:"+value( sum(P_sell) / (sum(P_PV + P_WP)) )*100+"%" )
disp("购电率:"+value( sum(P_buy) / (sum(P_PV + P_WP)) )*100+"%" )

disp("售氨收益:"+value(E_NH3))
disp("系统总收益:"+value(E))


% 氨平准化成本
L_COA = (value(C_IO + C_loss + C_buy - E_sell))/value(sum(m_NH3));
disp("氨平准化成本(元/t):"+L_COA*1000);

%% 画图分析 TODO 
figure
tiledlayout(2,1);
nexttile
% 功率平衡示意图
% --- 先用 value() 函数获取所有变量的数值解 ---
P_WP_val = value(P_WP);
P_PV_val = value(P_PV);
P_H2_out_val = value(P_H2_out);
P_NH3_out_val = value(P_NH3_out);
P_buy_val = value(P_buy);
P_loss_val = value(P_loss);
P_H2_in_val = value(P_H2_in);
P_N2_val = value(P_N2);
P_NH3_in_val = value(P_NH3_in);
P_sell_val = value(P_sell);

yyaxis left
plot_data = [P_WP_val/1000, P_PV_val/1000, P_H2_out_val/1000, P_NH3_out_val/1000, P_buy_val/1000, ...
             -P_loss_val/1000, -P_H2_in_val/1000, -P_N2_val/1000, -P_NH3_in_val/1000, -P_sell_val/1000];

% 先用默认方式绘图，并获取图形对象句柄 h
h = bar(0:T, plot_data, 'stacked');

% 使用 lines 颜色表生成差距较大的颜色
num_series = size(plot_data, 2);
colors = lines(num_series);

% 循环为每一个数据序列（条形）设置颜色
for i = 1:num_series
    h(i).FaceColor = colors(i, :);
end

% ... 后续的 title, legend, ylabel 等代码 ...

title('功率平衡图');
ylabel('出力及负荷/MW');
xlabel('时间/h');

yyaxis right
plot(0:T, c_pe,'LineWidth',1,'Color',[0 0 0]); % 电价
ylabel('电价/(元/kWh)');
legend('风电','光伏','氢燃料电池','氨燃料电池','大电网购电', ...
       '弃电','电解槽','空分制氮','H-B制氨','大电网售电','电价', 'Location', 'eastoutside'); % 建议加上位置参数以防遮挡


% 零刻线对齐，显示更方便
% 获取左右y轴范围
yyaxis left;
ylim_left = ylim;  % 左侧y轴当前范围
yyaxis right;
ylim_right = ylim; % 右侧y轴当前范围

% 计算左侧y轴零刻度的归一化位置
pos_zero = (0 - ylim_left(1)) / (ylim_left(2) - ylim_left(1));

% 调整右侧y轴范围，使零刻度位置对齐
if pos_zero >= 0 && pos_zero <= 1
    % 根据左侧零刻度位置计算右侧新范围
    new_ymin = (0 - ylim_right(2) * pos_zero) / (1 - pos_zero);
    ylim_right_new = [new_ymin, ylim_right(2)];
else
    % 若零刻度在左侧轴范围外，保持右侧轴原范围
    ylim_right_new = ylim_right;
end

% 应用新范围
yyaxis right;
ylim(ylim_right_new);


nexttile
% 氢平衡示意图，折线为氢储罐
yyaxis left
plot_data = [value(m_H2_in)/1000,-value(m_H2_NH3)/1000,-value(m_H2_fuel)/1000];
h = bar(0:T,plot_data,'stacked');
title('氢平衡图');
xlabel("时间/h");
ylabel("氢负荷/t");
% 使用 lines 颜色表生成差距较大的颜色
num_series = size(plot_data, 2);
colors = lines(num_series);

% 循环为每一个数据序列（条形）设置颜色
for i = 1:num_series
    h(i).FaceColor = colors(i, :);
end
yyaxis right
plot(0:T, value(L_H2)/1000,'LineWidth',1,'Color',[0 0 0]); % 储氢罐调度
ylabel('储氢罐载量/t');

legend("制氢","合成氨耗氢","燃料耗氢","储氢罐调度", 'Location', 'eastoutside');


%% 物料平衡验证
if sol.problem == 0
    % 检查氢气平衡
    H2_produced = value(sum(m_H2_in));
    H2_used_NH3 = value(sum(m_H2_NH3));
    H2_used_fuel = value(sum(m_H2_fuel));
    fprintf('氢气平衡: 生产=%.2f kg, 制氨用=%.2f kg, 燃料电池用=%.2f kg\n',...
        H2_produced, H2_used_NH3, H2_used_fuel);
    
    % 检查氨平衡
    NH3_produced = value(sum(m_NH3_in));
    NH3_sold = value(sum(m_NH3));
    NH3_fuel = value(sum(m_NH3_fuel));
    fprintf('氨平衡: 生产=%.2f kg, 售出=%.2f kg, 燃料电池用=%.2f kg\n',...
        NH3_produced, NH3_sold, NH3_fuel);
    
    % 检查储罐变化
    fprintf('储氢罐变化: 初始=%.2f kg, 最终=%.2f kg\n',...
        value(L_H2(1)), value(L_H2(end)));
    fprintf('储氨罐变化: 初始=%.2f kg, 最终=%.2f kg\n',...
        value(L_NH3(1)), value(L_NH3(end)));
end
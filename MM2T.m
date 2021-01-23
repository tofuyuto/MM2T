function MM2T(Global)
% <algorithm> <EMMO>
% Multi-objective Mult-modal to Two-objective 
% type --- 3 --- The type of aggregation function
% off --- 1 --- The type of offspring generation (1 = 'SBX') , (2 = 'DE')
% S --- 10 --- Subpopulation size

%------------------------------- Reference --------------------------------
% Takafumi Fukase, Naoki Masuyama, Yuusuke Nojima, Hisao Ishibuchi, 
% A Constrained Multi objective Evolutionary Algorithm Based on 
% Transformation to Two objective Optimization Problems, In Proc. of
% FAN2019, Toyama, 2019.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [type,S,seed] = Global.ParameterSet(3,10,Global.GetObj.run);
    rng('default');
    rng(seed);
    %{
    % seed値を固定したくなければ以下
    [type,S] = Global.ParameterSet(3,10);
    %}
    
    %% Generate the weight vectors
    [W,Global.N] = UniformPoint(Global.N,Global.M);
    
    %% Generate random population
    Population = Global.Initialization();
    Nadir = max(Population.objs,[],1);
    Z = min(Population.objs,[],1);
    
    %% 初期個体を Global.N 個の初期サブ個体群に収容
    UpdatePopulation = Population;
    minusNich = decide_nich(Population.decs);
    for i = 1 : Global.N
        Scalarizing_value = MM2TScalarizingFunc(type,W(i,:),Population.objs,Z,Nadir);
        % スカラー化関数と制約違反量を目的関数値とする個体群を作成．
        [SubPopulation(i,:),SubPop_Niching_value(i,:),SubPop_Scalarizing_value(i,:),FrontNo(i,:),CrowdDis(i,:)] = EnvironmentalSelectionMM2T(UpdatePopulation,minusNich,Scalarizing_value,S);
    end
    Offspring = Population;
    
    %% Optimization
    while Global.NotTermination(Population)
        Nadir = Z;
        % サブ個体群内で子個体の生成と参照点の更新．
        Offspring_gennum = 1;
        I = randperm(Global.N);
        for i = 1 : Global.N
            % Generate an offspring
            MatingPool = TournamentSelection(2,2,SubPop_Scalarizing_value(I(i),:),SubPop_Niching_value(I(i),:));
            Offspring(Offspring_gennum) = GAhalf(SubPopulation(I(i),MatingPool));
            Nadir = max(Nadir,Offspring(Offspring_gennum).obj); 
            Z = min(Z,Offspring(Offspring_gennum).obj);
            Offspring_gennum = Offspring_gennum + 1;
        end
        
        % 現在の現個体群と全ての子個体群を合わせた Gloal.N + S 個でサブ個体群の更新を行う．
        Offspring_Objs = Offspring.objs;       
        for i = 1 : Global.N
            % 子個体に目的関数（スカラー化関数と制約違反量）をセット．
            Curent_Objs = SubPopulation(i,:).objs;
            Offspring_Scalarizing_value = MM2TScalarizingFunc(type,W(i,:),Offspring_Objs,Z,Nadir);
            Curent_Scalarizing_value = MM2TScalarizingFunc(type,W(i,:),Curent_Objs,Z,Nadir);
            Update_minusNiching = decide_nich([SubPopulation(i,:).decs; Offspring.decs]);
            Update_Scalarizing_value = [Curent_Scalarizing_value; Offspring_Scalarizing_value];
            
            % Gloal.N + S 個でサブ個体群 i の更新．
            [SubPopulation(i,:),SubPop_Niching_value(i,:),SubPop_Scalarizing_value(i,:),FrontNo(i,:),CrowdDis(i,:)] = EnvironmentalSelectionMM2T([SubPopulation(i,:),Offspring],Update_minusNiching,Update_Scalarizing_value,S);
        end
        
        
        %% Output
        
        % サブ個体群を1つの個体群に集約．
        % 出力される個体群サイズは，Global.NではなくGlobal.N*Sであることに注意する．
        for i = 1 : Global.N
            Population(S*(i-1)+1:S*i) = SubPopulation(i,:);
        end
        
        %{
        % サブ個体群から代表点を取り出す．
        % 出力される個体群サイズは，Global.N．
        for i = 1 : Global.N
            Population(i) = SubPopulation(i,TournamentSelection(S,1,SubPop_Niching_value(i,:),SubPop_Scalarizing_value(i,:)));
        end
        %}
    end
end

function Nich = decide_nich(PopDec)
    N = size(PopDec,1);
    nb = min(3,max(N-1,0));
    
    d_dec = pdist2(PopDec,PopDec,'euclidean');
    temp = sort(d_dec);
    
    sigma_dec = sum(sum(temp(1:nb+1,:)))./(nb.*N);
    
    if sigma_dec == 0
        sigma_dec = inf;
    end
    
    neighbor_dec = d_dec < sigma_dec;
    Sh_dec = 1-d_dec./sigma_dec-eye(N);
    
    Nich = sum(Sh_dec.*neighbor_dec);
    Nich = transpose(Nich);
end


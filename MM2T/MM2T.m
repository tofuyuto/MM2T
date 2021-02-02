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
    % seed�l���Œ肵�����Ȃ���Έȉ�
    [type,S] = Global.ParameterSet(3,10);
    %}
    
    %% Generate the weight vectors
    [W,Global.N] = UniformPoint(Global.N,Global.M);
    
    %% Generate random population
    Population = Global.Initialization();
    Nadir = max(Population.objs,[],1);
    Z = min(Population.objs,[],1);
    
    %% �����̂� Global.N �̏����T�u�̌Q�Ɏ��e
    UpdatePopulation = Population;
    minusNich = decide_nich(Population.decs);
    for i = 1 : Global.N
        Scalarizing_value = MM2TScalarizingFunc(type,W(i,:),Population.objs,Z,Nadir);
        % �X�J���[���֐��Ɛ���ᔽ�ʂ�ړI�֐��l�Ƃ���̌Q���쐬�D
        [SubPopulation(i,:),SubPop_Niching_value(i,:),SubPop_Scalarizing_value(i,:),FrontNo(i,:),CrowdDis(i,:)] = EnvironmentalSelectionMM2T(UpdatePopulation,minusNich,Scalarizing_value,S);
    end
    Offspring = Population;
    
    %% Optimization
    while Global.NotTermination(Population)
        Nadir = Z;
        % �T�u�̌Q���Ŏq�̂̐����ƎQ�Ɠ_�̍X�V�D
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
        
        % ���݂̌��̌Q�ƑS�Ă̎q�̌Q�����킹�� Gloal.N + S �ŃT�u�̌Q�̍X�V���s���D
        Offspring_Objs = Offspring.objs;       
        for i = 1 : Global.N
            % �q�̂ɖړI�֐��i�X�J���[���֐��Ɛ���ᔽ�ʁj���Z�b�g�D
            Curent_Objs = SubPopulation(i,:).objs;
            Offspring_Scalarizing_value = MM2TScalarizingFunc(type,W(i,:),Offspring_Objs,Z,Nadir);
            Curent_Scalarizing_value = MM2TScalarizingFunc(type,W(i,:),Curent_Objs,Z,Nadir);
            Update_minusNiching = decide_nich([SubPopulation(i,:).decs; Offspring.decs]);
            Update_Scalarizing_value = [Curent_Scalarizing_value; Offspring_Scalarizing_value];
            
            % Gloal.N + S �ŃT�u�̌Q i �̍X�V�D
            [SubPopulation(i,:),SubPop_Niching_value(i,:),SubPop_Scalarizing_value(i,:),FrontNo(i,:),CrowdDis(i,:)] = EnvironmentalSelectionMM2T([SubPopulation(i,:),Offspring],Update_minusNiching,Update_Scalarizing_value,S);
        end
        
        
        %% Output
        
        % �T�u�̌Q��1�̌̌Q�ɏW��D
        % �o�͂����̌Q�T�C�Y�́CGlobal.N�ł͂Ȃ�Global.N*S�ł��邱�Ƃɒ��ӂ���D
        for i = 1 : Global.N
            Population(S*(i-1)+1:S*i) = SubPopulation(i,:);
        end
        
        %{
        % �T�u�̌Q�����\�_�����o���D
        % �o�͂����̌Q�T�C�Y�́CGlobal.N�D
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


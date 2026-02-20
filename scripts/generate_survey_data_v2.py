# -*- coding: utf-8 -*-
"""
RPA技术在金融经济领域应用需求调研问卷 - 模拟数据生成器
版本: v2.0
生成份数: 500份
输出格式: CSV
修复: Q12_RD(风险感知)和Q13_RV(收益感知)变量映射错误
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random
import os

np.random.seed(42)
random.seed(42)

N_SAMPLES = 500

INDUSTRIES = {
    "银行业": ["公司金融部", "零售银行部", "普惠金融部", "风险管理部", "运营管理部", "科技金融部", "私人银行部", "信用卡中心", "其他部门"],
    "证券业": ["投资银行部", "研究所", "资产管理部", "经纪业务部", "自营业务部", "财富管理部", "其他部门"],
    "保险业": ["理赔部", "核保部", "精算部", "产品开发部", "客户服务部", "其他部门"],
    "基金业": ["投资部", "研究部", "市场部", "运营部", "风控合规部", "其他部门"],
    "信托业": ["信托业务部", "资产管理部", "风控合规部", "其他部门"],
    "期货业": ["经纪业务部", "研究所", "风控部", "其他部门"],
    "金融租赁": ["业务部", "风控部", "运营部", "其他部门"],
    "金融科技": ["产品研发部", "技术部", "业务部", "其他部门"],
    "支付/第三方支付": ["业务部", "技术部", "风控部", "其他部门"],
    "财务公司": ["资金部", "财务部", "风控部", "其他部门"],
    "RPA/Agent厂商": ["产品研发/交付部", "模型算法部", "其他部门"],
    "AI技术/金融科技商": ["产品研发/交付部", "模型算法部", "其他部门"],
    "基础设施/云厂商": ["算力运维/SRE部", "其他部门"],
    "第三方数据商": ["数据合规与产品部", "其他部门"],
    "监管机构/审计": ["合规监管(RegTech)", "其他部门"],
    "系统集成商/咨询商": ["交付/实施中心", "其他部门"],
    "垂直SaaS商": ["产品生态部", "其他部门"],
    "高校/科研院所": ["产教融合中心", "其他部门"],
    "其他金融机构": ["其他部门"]
}

INDUSTRY_WEIGHTS = [0.20, 0.12, 0.08, 0.06, 0.04, 0.03, 0.03, 0.08, 0.05, 0.03, 0.05, 0.04, 0.03, 0.02, 0.02, 0.04, 0.02, 0.02, 0.04]

ECO_ROLES = [
    "应用端-作为RPA使用者，在业务场景中应用RPA提升效率",
    "供应端-作为RPA厂商，提供RPA产品和技术解决方案",
    "支撑端-作为云厂商/数据商，提供基础设施或数据服务",
    "治理端-作为监管机构，制定政策或进行合规审计",
    "交付端-作为集成商/咨询商，提供RPA实施交付服务",
    "环境端-作为SaaS商，提供垂直行业解决方案",
    "人才端-作为高校/培训机构，培养RPA专业人才"
]
ECO_ROLE_WEIGHTS = [0.45, 0.15, 0.08, 0.05, 0.12, 0.08, 0.07]

RPA_PRODUCTS = {
    "国际RPA厂商": ["UiPath", "Automation Anywhere", "Blue Prism", "微软Power Automate", "IBM RPA"],
    "国内RPA厂商": ["来也科技", "影刀RPA", "弘玑Cyclone", "云扩科技", "实在智能", "金智维", "艺赛旗"],
    "科技巨头": ["华为WeAutomate", "阿里云RPA", "腾讯云RPA", "百度RPA", "字节跳动RPA"],
    "AI Agent平台": ["Dify", "n8n", "LangChain", "AutoGPT", "Coze(扣子)"],
    "其他": ["自研产品", "其他"]
}

TECH_MATURITY_ITEMS = [
    "规则驱动流程自动化", "数据采集与录入", "报表生成与分发",
    "OCR文档识别", "NLP文本处理", "LLM大语言模型集成",
    "金融垂直大模型(Fin-LLM)", "智能风控模型", "反欺诈模型",
    "知识图谱构建", "数据中台", "数据安全技术",
    "国产化适配", "信创生态集成",
    "多模态数据处理", "流程挖掘与优化",
    "Agentic RPA(智能体自动化)", "自主决策与执行", "异常自修复能力",
    "智能运维(AIOps)", "云弹性伸缩", "Kubernetes容器编排"
]

PAIN_POINTS = [
    "数据孤岛", "非结构化数据", "数据质量差", "数据安全顾虑",
    "稳定性差", "系统兼容性问题", "脚本维护成本高", "AI融合困难", "技术淘汰快",
    "RPA人才短缺", "培训成本高",
    "实施成本高", "ROI不明确", "算力成本高",
    "合规风险担忧", "监管规则模糊",
    "业务流程不标准", "跨部门协作困难",
    "滞后性", "人眼极限"
]

DEMATEL_FACTORS = ["数据孤岛", "AI融合困难", "人才短缺", "实施成本高", "合规风险", "系统兼容性"]

BARRIERS = ["预算限制", "技术能力不足", "业务流程不标准", "数据质量问题", "管理层支持不足", 
            "员工抵触情绪", "安全合规顾虑", "ROI难以量化", "技术淘汰风险", "跨部门协作困难", 
            "信创适配困难", "AI融合技术门槛"]

TTF_ITEMS = ["RPA技术能力与业务任务需求匹配度高", "RPA能够有效处理贵机构的核心业务流程",
             "RPA技术适配金融行业的合规要求", "RPA与现有系统的集成能力强"]

SI_ITEMS = ["同行业企业普遍应用RPA，形成示范效应", "金融科技政策引导推动RPA应用",
            "市场竞争压力促使效率提升需求", "监管合规要求推动RPA需求"]

PV_ITEMS = ["RPA能显著提升工作效率(效率提升60%以上)", "RPA能有效降低运营成本(成本节约20%+)",
            "RPA能降低业务操作风险(错误率降至0.5%以下)", "RPA能提升合规水平，满足监管要求"]

RD_ITEMS = [
    "担心RPA技术不够成熟稳定，可能影响业务连续性",
    "担心数据安全和隐私泄露风险",
    "担心RPA应用可能带来合规风险",
    "担心RPA实施成本过高，ROI不明确",
    "担心RPA人才短缺，维护困难",
    "担心RPA实施过程影响现有业务运营",
    "担心RPA脚本维护成本持续增加",
    "担心RPA与现有系统集成困难"
]

RV_ITEMS = [
    "RPA能显著提升工作效率",
    "RPA能有效降低运营成本",
    "RPA能提升工作质量",
    "RPA能增强企业竞争力",
    "RPA能促进业务创新",
    "RPA能提升合规水平"
]

BI_ITEMS = [
    "未来愿意继续扩大RPA应用规模",
    "愿意向同行推荐RPA解决方案",
    "愿意增加RPA相关投资预算",
    "愿意参与RPA相关培训和推广"
]

RPA_DEMANDS = ["效率提升类", "风险管控类", "成本控制类", "数据处理类", 
               "业务自动化类", "体验优化类", "信创国产化", "AI智能增强"]

TECH_SOLUTIONS = [
    "LLM+RAG", "金融垂直大模型(Fin-LLM)", "多模态AI", "NLP自然语言处理", "OCR智能文档处理",
    "智能风控模型", "AI机器翻译",
    "知识图谱", "数据中台", "数据安全技术", "联邦学习",
    "流程挖掘", "低代码/无代码平台", "自修复算法",
    "云原生RPA", "Kubernetes容器编排", "云弹性伸缩",
    "国产化适配技术", "信创生态集成",
    "智能运维(AIOps)", "DevOps/CI/CD",
    "API网关与集成技术", "多源数据接口",
    "区块链存证与溯源",
    "Agentic RPA", "LangChain/Dify等Agent开发框架"
]

AHP_ITEMS = ["LLM+RAG", "多模态AI", "NLP", "知识图谱", "流程挖掘", "低代码平台", "云原生RPA", "数据安全技术", "DevOps"]

BUSINESS_SCENARIOS = {
    "财务金融": ["银企对账", "费用报销", "税务申报", "合并报表", "财务数据采集", "供应商对账", "ERP/财务系统/业务系统对接"],
    "风险管理": ["反洗钱调查", "合规报表", "征信查询", "风险预警", "内控审计", "反欺诈检测", "穿透式监控"],
    "信贷业务": ["信贷审批", "贷后管理", "抵押物管理", "征信数据采集", "贷款材料核验", "准入门槛审核"],
    "客户服务": ["智能客服", "工单处理", "客户信息维护", "营销自动化", "客户KYC", "精准营销"],
    "投研分析": ["研报摘要生成", "行情归因分析", "财报解析", "投资决策支持", "量化交易", "多维行情自动归因"],
    "数据治理": ["跨系统数据同步", "数据清洗", "数据质量监控", "数据安全审计", "数据要素流通", "隐私计算"],
    "保险业务": ["理赔审核", "核保自动化", "保单核实", "事故照片审核", "医疗险理赔决策", "车险理赔"],
    "政务金融": ["政务审批", "社保结算", "公积金管理", "公共数据融合", "智慧城市"],
    "运维支持": ["资源调度", "负载均衡", "故障诊断", "性能监控", "系统巡检", "GPU算力管理"],
    "人力资源": ["简历筛选", "员工管理", "培训管理", "薪酬核算"],
    "其他场景": ["报表自动化", "邮件处理", "流程监控", "其他"]
}

POLICY_ITEMS = ["金融科技政策推动了贵机构RPA应用", "监管合规要求增加了RPA应用需求",
                "信创政策(国产化要求)影响了RPA选型", "数据安全法规对RPA应用提出了更高要求",
                "数字金融政策为RPA发展提供了良好环境", "科技金融政策促进了RPA在创新业务中的应用"]

FUTURE_TRENDS = ["AI大模型深度集成，实现智能决策", "多模态数据处理能力大幅提升",
                 "自主学习和自我优化能力", "跨系统无缝集成能力",
                 "实时流程挖掘与优化", "自然语言交互式流程设计",
                 "智能异常处理与自修复", "端到端业务流程自动化",
                 "行业垂直解决方案成熟", "信创生态完善，国产替代加速",
                 "Agentic RPA(智能体RPA)成为主流"]

TALENT_NEEDS = ["专业培训课程", "厂商认证体系", "行业交流平台", "校企合作项目", 
                "内部培训指导", "人才招聘支持", "RPA项目实战经验培训", 
                "业务流程设计能力培训", "AI+RPA融合技术培训", "产教融合中心"]

PAIN_POINT_TEMPLATES = [
    "我们在RPA应用过程中遇到的最大痛点是数据孤岛问题。不同业务系统之间的数据无法有效互通，导致RPA机器人需要跨多个系统手动搬运数据，严重影响了自动化效率。同时，非结构化数据处理能力不足，很多业务文档如财报、合同等格式各异，OCR识别准确率不高，需要大量人工校验。",
    "最大的挑战是RPA与老旧核心系统的集成困难。我们的核心业务系统已有20多年历史，接口不开放，只能通过屏幕抓取方式操作，稳定性很差。每次系统升级，RPA脚本都需要重新调试，维护成本极高。",
    "AI融合是当前最大的痛点。虽然引入了大语言模型，但在金融专业场景下经常出现幻觉问题，无法准确理解复杂的金融术语和业务逻辑。同时，模型输出的可解释性差，难以满足监管合规要求。",
    "人才短缺问题非常严重。懂业务的人不懂技术，懂技术的人不懂金融业务。RPA开发需要既懂流程优化又懂编程的复合型人才，但市场上这样的人才非常稀缺，培养周期长，流失率高。",
    "实施成本和ROI难以量化是主要痛点。RPA项目前期投入大，包括软件授权、开发实施、培训维护等，但收益往往难以精确计算。很多项目实施后效果不如预期，导致管理层对后续投资持观望态度。",
    "合规风险是我们最担心的问题。RPA机器人操作涉及大量敏感数据，如何确保数据安全、操作可追溯、符合监管要求，目前还没有成熟的标准和解决方案。一旦出现数据泄露，后果不堪设想。",
    "跨部门协作困难是主要障碍。RPA项目往往涉及多个业务部门，但各部门利益诉求不同，流程标准化程度低，协调成本高。很多时候技术方案已经成熟，但因为部门间推诿扯皮，项目迟迟无法落地。",
    "技术更新迭代太快，投资风险高。去年采购的RPA平台，今年就有了更新的AI Agent方案。担心现有投资很快过时，又不敢贸然投入新技术，陷入两难境地。",
    "系统稳定性问题突出。RPA机器人在处理异常情况时经常卡住，需要人工干预。特别是涉及外部系统对接时，网络波动、页面加载慢等问题都会导致流程中断，影响业务连续性。",
    "培训成本高、学习曲线陡峭。RPA平台功能越来越复杂，普通业务人员难以掌握。专业开发人员又成本高昂，导致很多自动化需求无法及时响应，业务部门满意度低。"
]

DEMAND_TEMPLATES = [
    "希望未来RPA能够实现更深度的AI融合，特别是大语言模型与RPA的无缝集成。期望能够通过自然语言描述业务需求，系统自动生成自动化流程，大幅降低开发门槛。同时希望增强非结构化数据处理能力，能够准确识别各类金融文档。",
    "期待RPA向智能化、自主化方向发展。希望RPA机器人能够具备自我学习和优化能力，自动发现流程瓶颈并提出改进建议。同时希望能够实现端到端的业务流程自动化，而不仅仅是单个任务的自动化。",
    "希望RPA厂商能够提供更多金融行业垂直解决方案。目前很多方案都是通用的，对金融业务的特殊性考虑不足。期望有针对银行、证券、保险等细分领域的专业化解决方案，开箱即用。",
    "期望RPA能够更好地支持信创国产化。希望国产RPA产品能够快速成熟，与国产操作系统、数据库、中间件等实现良好适配，降低对国外产品的依赖，同时满足数据安全合规要求。",
    "希望RPA能够与数据中台、知识图谱等技术深度融合。通过构建企业级知识库，让RPA机器人能够理解业务上下文，做出更智能的决策。同时期望加强数据安全能力，支持隐私计算等新技术。",
    "期待RPA开发门槛进一步降低。希望低代码/无代码平台能够更加成熟，让业务人员也能快速构建自动化流程。同时希望厂商提供更完善的培训体系和社区支持，加速人才培养。",
    "希望RPA能够支持更复杂的业务场景。目前RPA主要处理规则明确的流程，期望未来能够处理更多需要判断和决策的场景，如风险评估、合规审查等，真正实现智能化运营。",
    "期待RPA运维能力提升。希望能够实现RPA机器人的智能监控、自动告警、故障自愈。同时期望云原生RPA能够更加成熟，支持弹性伸缩、按需付费，降低中小企业使用门槛。",
    "希望RPA生态更加开放。期望能够有统一的标准和接口，让不同厂商的RPA产品能够互联互通。同时希望能够更好地集成各类AI能力，如OCR、NLP、知识图谱等，形成完整的智能化解决方案。",
    "期待RPA在监管合规领域发挥更大作用。希望RPA能够支持穿透式监管、实时合规监控等场景，帮助金融机构更好地满足监管要求。同时期望监管机构能够出台RPA应用指导规范，降低合规风险。"
]

def generate_base_latent_scores(n):
    tech_capability = np.random.normal(3.5, 0.8, n)
    tech_capability = np.clip(tech_capability, 1, 5)
    
    org_digital_level = np.random.choice([1, 2, 3, 4], n, p=[0.15, 0.35, 0.35, 0.15])
    rpa_experience = np.random.choice([1, 2, 3, 4], n, p=[0.20, 0.35, 0.30, 0.15])
    
    return tech_capability, org_digital_level, rpa_experience

def generate_q1():
    industry = np.random.choice(list(INDUSTRIES.keys()), p=INDUSTRY_WEIGHTS)
    department = np.random.choice(INDUSTRIES[industry])
    return f"{industry}|{department}"

def generate_q2():
    employee_scale = np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
    enterprise_type = np.random.choice([1, 2, 3, 4], p=[0.35, 0.40, 0.15, 0.10])
    digital_stage = np.random.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.35, 0.15])
    rpa_years = np.random.choice([1, 2, 3, 4], p=[0.20, 0.35, 0.30, 0.15])
    rpa_scale = np.random.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.20, 0.10])
    return employee_scale, enterprise_type, digital_stage, rpa_years, rpa_scale

def generate_q4():
    selected = []
    if np.random.random() < 0.25:
        selected.append(np.random.choice(RPA_PRODUCTS["国际RPA厂商"]))
    if np.random.random() < 0.45:
        selected.extend(np.random.choice(RPA_PRODUCTS["国内RPA厂商"], 
                                         size=min(np.random.randint(1, 3), 7), replace=False).tolist())
    if np.random.random() < 0.20:
        selected.append(np.random.choice(RPA_PRODUCTS["科技巨头"]))
    if np.random.random() < 0.15:
        selected.append(np.random.choice(RPA_PRODUCTS["AI Agent平台"]))
    if np.random.random() < 0.10:
        selected.append(np.random.choice(RPA_PRODUCTS["其他"]))
    return "|".join(selected) if selected else "未使用"

def generate_likert_with_base(base, n_items, noise=0.6):
    scores = np.random.normal(base.reshape(-1, 1), noise, (len(base), n_items))
    return np.clip(scores, 1, 5).astype(int)

def generate_q7_dematel():
    matrix = np.zeros((6, 6), dtype=int)
    for i in range(6):
        for j in range(6):
            if i != j:
                if i in [0, 1, 5]:
                    matrix[i, j] = np.random.choice([0, 1, 2, 3], p=[0.15, 0.25, 0.35, 0.25])
                else:
                    matrix[i, j] = np.random.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15])
    return matrix

def generate_q8_ranking():
    weights = np.random.dirichlet(np.ones(12))
    ranks = np.argsort(-weights) + 1
    return ranks

def generate_q14_fuzzy():
    results = {}
    for item in TECH_SOLUTIONS:
        most_likely = np.random.uniform(5, 9)
        lower = most_likely - np.random.uniform(1, 2)
        upper = most_likely + np.random.uniform(1, 2)
        results[item] = (max(1, round(lower, 1)), round(most_likely, 1), min(10, round(upper, 1)))
    return results

def generate_q15_ahp():
    n = len(AHP_ITEMS)
    weights = np.random.dirichlet(np.ones(n) * 2)
    comparisons = []
    for i in range(n):
        for j in range(i + 1, n):
            ratio = weights[i] / weights[j] if weights[j] > 0 else 1
            if ratio > 1:
                saaty = min(9, max(1, round(ratio * 2)))
                comparisons.append((AHP_ITEMS[i], saaty, 0, AHP_ITEMS[j]))
            else:
                saaty = min(9, max(1, round((1/ratio) * 2)))
                comparisons.append((AHP_ITEMS[i], 0, saaty, AHP_ITEMS[j]))
    return comparisons

def generate_q16_scenarios():
    selected = []
    for category, items in BUSINESS_SCENARIOS.items():
        if np.random.random() < 0.5:
            n_select = np.random.randint(1, min(4, len(items) + 1))
            chosen = np.random.choice(items, size=n_select, replace=False).tolist()
            selected.extend([f"{category}|{item}" for item in chosen])
    return "|".join(selected) if selected else "未部署"

def generate_q17_discrete():
    efficiency = np.random.choice([1, 2, 3, 4], p=[0.15, 0.30, 0.35, 0.20])
    cost_save = np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
    error_reduce = np.random.choice([1, 2, 3, 4], p=[0.20, 0.30, 0.30, 0.20])
    satisfaction = np.random.choice([1, 2, 3, 4], p=[0.10, 0.25, 0.40, 0.25])
    compliance = np.random.choice([1, 2, 3, 4], p=[0.10, 0.20, 0.40, 0.30])
    roi = np.random.choice([1, 2, 3, 4], p=[0.15, 0.30, 0.35, 0.20])
    return efficiency, cost_save, error_reduce, satisfaction, compliance, roi

def generate_open_text(templates, min_len=100):
    base = np.random.choice(templates)
    if len(base) < min_len:
        additions = [
            "此外，我们还希望能够有更完善的技术支持和售后服务。",
            "同时，期待行业内有更多的交流和分享平台。",
            "希望相关标准和规范能够尽快完善，指导行业健康发展。",
            "建议加强产学研合作，推动技术创新和人才培养。"
        ]
        base = base + np.random.choice(additions)
    return base

def generate_dataset():
    tech_cap, org_digital, rpa_exp = generate_base_latent_scores(N_SAMPLES)
    
    ttf_base = 2.5 + 0.3 * tech_cap + 0.2 * org_digital
    si_base = 2.8 + 0.15 * org_digital
    pv_base = 2.6 + 0.25 * tech_cap + 0.2 * rpa_exp
    rd_base = 3.5 - 0.1 * tech_cap + 0.15 * (5 - org_digital)
    rv_base = 2.8 + 0.2 * tech_cap + 0.15 * rpa_exp
    bi_base = 2.5 + 0.3 * pv_base - 0.15 * rd_base + 0.1 * ttf_base
    
    data = []
    
    for i in range(N_SAMPLES):
        record = {}
        record['respondent_id'] = f"R{str(i+1).zfill(4)}"
        record['submit_time'] = f"2026-02-{np.random.randint(10, 20):02d} {np.random.randint(8, 18):02d}:{np.random.randint(0, 60):02d}:{np.random.randint(0, 60):02d}"
        
        q1 = generate_q1()
        record['Q1_industry'] = q1.split('|')[0]
        record['Q1_department'] = q1.split('|')[1]
        
        q2 = generate_q2()
        record['Q2_employee_scale'] = q2[0]
        record['Q2_enterprise_type'] = q2[1]
        record['Q2_digital_stage'] = q2[2]
        record['Q2_rpa_years'] = q2[3]
        record['Q2_rpa_scale'] = q2[4]
        
        record['Q3_eco_role'] = np.random.choice(ECO_ROLES, p=ECO_ROLE_WEIGHTS)
        record['Q4_products'] = generate_q4()
        
        q5_scores = generate_likert_with_base(np.array([tech_cap[i]]), len(TECH_MATURITY_ITEMS))
        for j, item in enumerate(TECH_MATURITY_ITEMS):
            record[f'Q5_{j+1}'] = q5_scores[0, j]
        
        q6_scores = generate_likert_with_base(np.array([4 - tech_cap[i]]), len(PAIN_POINTS), noise=0.7)
        for j, item in enumerate(PAIN_POINTS):
            record[f'Q6_{j+1}'] = q6_scores[0, j]
        
        q7_matrix = generate_q7_dematel()
        for row in range(6):
            for col in range(6):
                if row != col:
                    record[f'Q7_{row+1}_{col+1}'] = q7_matrix[row, col]
        
        q8_ranks = generate_q8_ranking()
        for j, rank in enumerate(q8_ranks):
            record[f'Q8_{j+1}_rank'] = rank
        
        q9_scores = generate_likert_with_base(np.array([ttf_base[i]]), len(TTF_ITEMS))
        for j, item in enumerate(TTF_ITEMS):
            record[f'Q9_TTF{j+1}'] = q9_scores[0, j]
        
        q10_scores = generate_likert_with_base(np.array([si_base[i]]), len(SI_ITEMS))
        for j, item in enumerate(SI_ITEMS):
            record[f'Q10_SI{j+1}'] = q10_scores[0, j]
        
        q11_scores = generate_likert_with_base(np.array([pv_base[i]]), len(PV_ITEMS))
        for j, item in enumerate(PV_ITEMS):
            record[f'Q11_PV{j+1}'] = q11_scores[0, j]
        
        q12_scores = generate_likert_with_base(np.array([rd_base[i]]), len(RD_ITEMS))
        for j, item in enumerate(RD_ITEMS):
            record[f'Q12_RD{j+1}'] = q12_scores[0, j]
        
        q13_scores = generate_likert_with_base(np.array([rv_base[i]]), len(RV_ITEMS))
        for j, item in enumerate(RV_ITEMS):
            record[f'Q13_RV{j+1}'] = q13_scores[0, j]
        
        q14_scores = generate_likert_with_base(np.array([bi_base[i]]), len(BI_ITEMS))
        for j, item in enumerate(BI_ITEMS):
            record[f'Q14_BI{j+1}'] = q14_scores[0, j]
        
        q15_fuzzy = generate_q14_fuzzy()
        tech_solutions = list(TECH_SOLUTIONS)
        for j, item in enumerate(tech_solutions):
            low, mid, high = q15_fuzzy[item]
            record[f'Q15_{j+1}_low'] = low
            record[f'Q15_{j+1}_mid'] = mid
            record[f'Q15_{j+1}_high'] = high
        
        q16_comparisons = generate_q15_ahp()
        for j, (a, a_val, b_val, b) in enumerate(q16_comparisons):
            record[f'Q16_{j+1}_A'] = a
            record[f'Q16_{j+1}_A_val'] = a_val
            record[f'Q16_{j+1}_B_val'] = b_val
            record[f'Q16_{j+1}_B'] = b
        
        record['Q17_scenarios'] = generate_q16_scenarios()
        
        q18 = generate_q17_discrete()
        record['Q18_efficiency'] = q18[0]
        record['Q18_cost_save'] = q18[1]
        record['Q18_error_reduce'] = q18[2]
        record['Q18_satisfaction'] = q18[3]
        record['Q18_compliance'] = q18[4]
        record['Q18_roi'] = q18[5]
        
        q19_scores = generate_likert_with_base(np.array([si_base[i]]), len(POLICY_ITEMS))
        for j, item in enumerate(POLICY_ITEMS):
            record[f'Q19_{j+1}'] = q19_scores[0, j]
        
        n_trends = np.random.randint(3, 8)
        record['Q20_trends'] = "|".join(np.random.choice(FUTURE_TRENDS, n_trends, replace=False))
        
        n_talents = np.random.randint(3, 7)
        record['Q21_talents'] = "|".join(np.random.choice(TALENT_NEEDS, n_talents, replace=False))
        
        record['Q22_pain_point_text'] = generate_open_text(PAIN_POINT_TEMPLATES)
        record['Q23_demand_text'] = generate_open_text(DEMAND_TEMPLATES)
        
        data.append(record)
    
    return pd.DataFrame(data)

def main():
    print("=" * 60)
    print("RPA技术在金融经济领域应用需求调研问卷 - 模拟数据生成器 v2.0")
    print("=" * 60)
    print(f"生成份数: {N_SAMPLES}")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    df = generate_dataset()
    
    output_dir = "e:\\B正大杯\\dataexample"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "survey_data_simulated.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"数据已保存至: {output_file}")
    print("-" * 60)
    print("数据概览:")
    print(f"  - 总样本数: {len(df)}")
    print(f"  - 总变量数: {len(df.columns)}")
    print(f"  - 数据完整性: {df.isnull().sum().sum()} 个缺失值")
    print("-" * 60)
    print("SEM核心变量统计:")
    print(f"  - Q9 TTF均值: {df[[c for c in df.columns if c.startswith('Q9_TTF')]].mean().mean():.2f}")
    print(f"  - Q10 SI均值: {df[[c for c in df.columns if c.startswith('Q10_SI')]].mean().mean():.2f}")
    print(f"  - Q11 PV均值: {df[[c for c in df.columns if c.startswith('Q11_PV')]].mean().mean():.2f}")
    print(f"  - Q12 RD均值: {df[[c for c in df.columns if c.startswith('Q12_RD')]].mean().mean():.2f}")
    print(f"  - Q13 RV均值: {df[[c for c in df.columns if c.startswith('Q13_RV')]].mean().mean():.2f}")
    print(f"  - Q14 BI均值: {df[[c for c in df.columns if c.startswith('Q14_BI')]].mean().mean():.2f}")
    print("=" * 60)
    print("模拟数据生成完成！")

if __name__ == "__main__":
    main()

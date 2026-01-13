'use client'

import { useState, useEffect } from 'react'
import { motion, useScroll, useSpring } from 'framer-motion'
import { 
  Cpu, Database, Search, Merge, Download, 
  Building2, Brain, Landmark, Stethoscope, 
  GraduationCap, Bot, Target, TrendingUp, 
  Shield, Globe, Rocket, ChevronDown, Menu, X,
  ArrowRight, CheckCircle, Sparkles, Zap,
  FileText, FileAudio, FileVideo, Code, Database as DbIcon
} from 'lucide-react'

// Navigation Component
function Navigation() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const navLinks = [
    { name: 'Vision', href: '#vision' },
    { name: 'Product', href: '#product' },
    { name: 'Components', href: '#components' },
    { name: 'Customers', href: '#customers' },
    { name: 'Use Cases', href: '#usecases' },
    { name: 'Market', href: '#market' },
    { name: 'GTM', href: '#gtm' },
  ]

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled ? 'glass py-4' : 'bg-transparent py-6'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
        <a href="#" className="flex items-center space-x-2">
          <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
            <Cpu className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-bold font-display">KERNELIZE</span>
        </a>

        {/* Desktop Navigation */}
        <div className="hidden lg:flex items-center space-x-8">
          {navLinks.map((link) => (
            <a
              key={link.name}
              href={link.href}
              className="text-gray-300 hover:text-white transition-colors text-sm font-medium"
            >
              {link.name}
            </a>
          ))}
          <button className="btn-primary text-sm">
            Get Early Access
          </button>
        </div>

        {/* Mobile Menu Button */}
        <button
          className="lg:hidden text-white"
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        >
          {isMobileMenuOpen ? <X /> : <Menu />}
        </button>
      </div>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="lg:hidden absolute top-full left-0 right-0 glass border-t border-white/10 py-4"
        >
          <div className="flex flex-col space-y-4 px-6">
            {navLinks.map((link) => (
              <a
                key={link.name}
                href={link.href}
                className="text-gray-300 hover:text-white transition-colors"
                onClick={() => setIsMobileMenuOpen(false)}
              >
                {link.name}
              </a>
            ))}
            <button className="btn-primary w-full">
              Get Early Access
            </button>
          </div>
        </motion.div>
      )}
    </motion.nav>
  )
}

// Hero Section
function Hero() {
  const { scrollY } = useScroll()
  const y = useSpring(scrollY, { stiffness: 100, damping: 30 })

  return (
    <section className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="particle absolute"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 6}s`,
            }}
          />
        ))}
      </div>

      {/* Gradient Orbs */}
      <motion.div
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.3, 0.5, 0.3],
        }}
        transition={{ duration: 8, repeat: Infinity }}
        className="absolute top-1/4 -left-32 w-96 h-96 bg-primary/20 rounded-full blur-3xl"
      />
      <motion.div
        animate={{
          scale: [1, 1.3, 1],
          opacity: [0.2, 0.4, 0.2],
        }}
        transition={{ duration: 10, repeat: Infinity, delay: 1 }}
        className="absolute bottom-1/4 -right-32 w-96 h-96 bg-secondary/20 rounded-full blur-3xl"
      />

      <div className="relative z-10 text-center px-6 max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center space-x-2 bg-white/5 border border-white/10 rounded-full px-4 py-2 mb-8">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm text-gray-300">The Future of Knowledge Infrastructure</span>
          </div>

          <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold mb-6">
            <span className="block">Compress the World's</span>
            <span className="gradient-text">Knowledge Into</span>
            <span className="block">Intelligence Kernels</span>
          </h1>

          <p className="text-xl md:text-2xl text-gray-400 mb-10 max-w-3xl mx-auto leading-relaxed">
            Build the world's first Knowledge Compression Infrastructure. 
            Enable software, AIs, and agents to store, search, and reason over 
            100×–10,000× compressed knowledge blocks without losing meaning.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button className="btn-primary flex items-center space-x-2 text-lg px-8 py-4">
              <span>Start Compressing</span>
              <ArrowRight className="w-5 h-5" />
            </button>
            <button className="btn-secondary flex items-center space-x-2 text-lg px-8 py-4">
              <span>View Documentation</span>
            </button>
          </div>

          {/* Stats */}
          <div className="mt-16 grid grid-cols-3 gap-8 max-w-3xl mx-auto">
            {[
              { value: '100×', label: 'Compression' },
              { value: '$3T+', label: 'Market Size' },
              { value: '10,000×', label: 'Max Ratio' },
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold gradient-text">{stat.value}</div>
                <div className="text-gray-500 text-sm mt-1">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      <motion.div
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
        className="absolute bottom-10 left-1/2 transform -translate-x-1/2 text-gray-500"
      >
        <ChevronDown className="w-6 h-6" />
      </motion.div>
    </section>
  )
}

// Vision Section
function Vision() {
  const comparisons = [
    { company: 'NVIDIA', role: 'GPU Infrastructure', kernelize: 'Semantic Compression' },
    { company: 'Snowflake', role: 'Data Cloud', kernelize: 'Knowledge Cloud' },
    { company: 'OpenAI', role: 'Intelligence Compute', kernelize: 'Intelligence Format' },
  ]

  return (
    <section id="vision" className="section relative">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="section-title">Our Vision</h2>
          <p className="section-subtitle mx-auto">
            Become the semantic compression layer of the entire digital world
          </p>
        </motion.div>

        {/* Comparison Table */}
        <div className="glass rounded-2xl p-8 mb-16">
          <div className="grid grid-cols-3 gap-8">
            {comparisons.map((row, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-gray-400 text-sm mb-2">{row.company}</div>
                <div className="text-white font-semibold mb-1">{row.role}</div>
                <div className="text-primary text-sm">↓</div>
                <div className="text-accent font-semibold mt-1">{row.kernelize}</div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Core Idea */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <h3 className="text-2xl md:text-3xl font-bold mb-8">The Core Idea</h3>
          <p className="text-gray-300 text-lg leading-relaxed max-w-4xl mx-auto mb-12">
            Every file, every book, every dataset, every video, every codebase, every policy → 
            compressed into tiny semantic "kernels" that preserve <span className="text-primary font-semibold">meaning</span>, 
            <span className="text-secondary font-semibold"> causality</span>, 
            <span className="text-accent font-semibold"> relationships</span>, 
            <span className="text-primary font-semibold"> context</span>, 
            <span className="text-secondary font-semibold"> domain expertise</span>, and 
            <span className="text-accent font-semibold"> reasoning patterns</span>.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {[
              { icon: Target, label: 'Meaning', color: 'primary' },
              { icon: TrendingUp, label: 'Causality', color: 'secondary' },
              { icon: Database, label: 'Relationships', color: 'accent' },
              { icon: Brain, label: 'Context', color: 'primary' },
              { icon: GraduationCap, label: 'Domain', color: 'secondary' },
              { icon: Bot, label: 'Reasoning', color: 'accent' },
            ].map((item, index) => (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className="glass rounded-xl p-4 text-center card-hover"
              >
                <item.icon className={`w-8 h-8 mx-auto mb-3 text-${item.color}`} />
                <div className="text-sm font-medium">{item.label}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}

// Product Section
function Product() {
  const features = [
    { icon: FileText, label: 'Text', desc: 'Documents, reports, articles' },
    { icon: FileText, label: 'PDFs', desc: 'Books, papers, forms' },
    { icon: FileAudio, label: 'Audio', desc: 'Podcasts, meetings' },
    { icon: FileVideo, label: 'Video', desc: 'Transcripts, lectures' },
    { icon: Code, label: 'Code', desc: 'Repositories, snippets' },
    { icon: DbIcon, label: 'Databases', desc: 'Structured data' },
  ]

  return (
    <section id="product" className="section bg-surface/50">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="section-title">Flagship Product: Kernel Engine</h2>
          <p className="section-subtitle mx-auto">
            A cloud platform that converts raw knowledge into Semantic Intelligence Kernels
          </p>
        </motion.div>

        {/* Feature Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6 mb-12">
          {features.map((feature, index) => (
            <motion.div
              key={feature.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
              className="glass rounded-2xl p-6 card-hover"
            >
              <feature.icon className="w-12 h-12 text-primary mb-4" />
              <h3 className="text-xl font-semibold mb-2">{feature.label}</h3>
              <p className="text-gray-400">{feature.desc}</p>
            </motion.div>
          ))}
        </div>

        {/* Value Proposition */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass rounded-2xl p-8 text-center"
        >
          <h3 className="text-2xl font-bold mb-6">Why Intelligence Kernels?</h3>
          <div className="grid md:grid-cols-4 gap-8">
            {[
              { value: 'Cheaper', desc: 'Reduced storage & compute costs' },
              { value: 'Faster', desc: 'Instant semantic search' },
              { value: 'Smaller', desc: '100×-10,000× compression' },
              { value: 'Accurate', desc: 'Preserved meaning & context' },
            ].map((item, index) => (
              <div key={item.value}>
                <div className="text-3xl font-bold gradient-text mb-2">{item.value}</div>
                <div className="text-gray-400 text-sm">{item.desc}</div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}

// Components Section
function Components() {
  const engines = [
    {
      id: 'compression',
      icon: Cpu,
      title: 'Compression Engine',
      desc: 'The core innovation. Compresses knowledge 100×–10,000× with semantic retention, causality preservation, and domain-knowledge preservation.',
      features: ['Semantic Retention', 'Causal Graphs', 'Conflict Resolution'],
    },
    {
      id: 'query',
      icon: Search,
      title: 'Query Engine',
      desc: 'Query compressed knowledge without expanding it. Works like: SELECT * FROM kernel WHERE meaning="climate change causes"',
      features: ['Sub-millisecond Search', 'Semantic Matching', 'No Decompression'],
    },
    {
      id: 'merge',
      icon: Merge,
      title: 'Merge Engine',
      desc: 'Allows incremental updating. Add new documents, modify old kernels, and recombine knowledge seamlessly.',
      features: ['Incremental Updates', 'Knowledge Recombination', 'Version Control'],
    },
    {
      id: 'distillation',
      icon: Download,
      title: 'Distillation Engine',
      desc: 'Inject kernels into LLMs, agents, AutoGPT-like systems, and inference devices like mobile and IoT.',
      features: ['LLM Integration', 'Agent Skill Packs', 'Edge Deployment'],
    },
    {
      id: 'index',
      icon: Database,
      title: 'Kernel Index',
      desc: 'Marketplace of domain kernels. Purchase pre-compressed knowledge for Law, Medicine, Finance, and more.',
      features: ['Domain Kernels', 'Training Data', 'Reasoning Models'],
    },
  ]

  const [activeEngine, setActiveEngine] = useState(0)

  return (
    <section id="components" className="section">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="section-title">Core Components</h2>
          <p className="section-subtitle mx-auto">
            Five powerful engines that power the Knowledge Compression Infrastructure
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-5 gap-8">
          {/* Engine Navigation */}
          <div className="lg:col-span-2 space-y-4">
            {engines.map((engine, index) => (
              <motion.button
                key={engine.id}
                onClick={() => setActiveEngine(index)}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className={`w-full text-left p-4 rounded-xl transition-all duration-300 ${
                  activeEngine === index
                    ? 'glass border-primary/50'
                    : 'bg-white/5 hover:bg-white/10'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <engine.icon className={`w-6 h-6 ${
                    activeEngine === index ? 'text-primary' : 'text-gray-400'
                  }`} />
                  <span className={`font-semibold ${
                    activeEngine === index ? 'text-white' : 'text-gray-400'
                  }`}>
                    {engine.title}
                  </span>
                </div>
              </motion.button>
            ))}
          </div>

          {/* Engine Details */}
          <motion.div
            key={activeEngine}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="lg:col-span-3 glass rounded-2xl p-8"
          >
            {(() => {
              const ActiveIcon = engines[activeEngine].icon;
              return <ActiveIcon className="w-16 h-16 text-primary mb-6" />;
            })()}
            <h3 className="text-2xl font-bold mb-4">{engines[activeEngine].title}</h3>
            <p className="text-gray-300 mb-6 leading-relaxed">
              {engines[activeEngine].desc}
            </p>
            <div className="flex flex-wrap gap-3">
              {engines[activeEngine].features.map((feature, index) => (
                <span
                  key={feature}
                  className="px-4 py-2 bg-primary/20 text-primary rounded-full text-sm"
                >
                  {feature}
                </span>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}

// Customers Section
function Customers() {
  const customers = [
    {
      icon: Building2,
      title: 'Enterprises',
      desc: '10-100TB of unstructured knowledge compressed to 50-500GB',
      metric: '500× Reduction',
    },
    {
      icon: Brain,
      title: 'AI Platforms',
      desc: 'Query kernels instead of raw tokens for 10× cheaper inference',
      metric: '10× Cheaper',
    },
    {
      icon: Landmark,
      title: 'Governments',
      desc: 'Laws, intelligence, archives, and decision support systems',
      metric: 'Secure & Fast',
    },
    {
      icon: Stethoscope,
      title: 'Healthcare',
      desc: 'Medical knowledge compressed for clinician support systems',
      metric: 'Life-Saving',
    },
    {
      icon: GraduationCap,
      title: 'Education',
      desc: 'Whole textbooks compressed to one kernel per subject',
      metric: '1000× Smaller',
    },
    {
      icon: Bot,
      title: 'Agent Platforms',
      desc: 'Download kernels as domain skill packs for autonomous agents',
      metric: 'Instant Skills',
    },
  ]

  return (
    <section id="customers" className="section bg-surface/50">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="section-title">Target Customers</h2>
          <p className="section-subtitle mx-auto">
            Organizations that need to store, search, and reason over massive knowledge bases
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {customers.map((customer, index) => (
            <motion.div
              key={customer.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
              className="glass rounded-2xl p-6 card-hover"
            >
              <customer.icon className="w-12 h-12 text-primary mb-4" />
              <h3 className="text-xl font-bold mb-2">{customer.title}</h3>
              <p className="text-gray-400 mb-4">{customer.desc}</p>
              <div className="text-2xl font-bold gradient-text">{customer.metric}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

// Use Cases Section
function UseCases() {
  const useCases = [
    {
      title: 'Company Knowledge Base',
      desc: 'Entire company knowledge → 1 compact kernel. Employees ask questions and get instant answers.',
      icon: Building2,
    },
    {
      title: 'LLM Training',
      desc: 'Feed kernels instead of raw documents → 10× cheaper fine-tuning with better retention.',
      icon: Brain,
    },
    {
      title: 'Policy Analysis',
      desc: 'Decades of government policies compressed → instant causal reasoning across regulations.',
      icon: Landmark,
    },
    {
      title: 'Anti-Misinformation',
      desc: 'Compare kernels to detect narrative gaps and inconsistencies across sources.',
      icon: Shield,
    },
    {
      title: 'Offline Intelligence',
      desc: 'Smartphones can store entire knowledge bases in just 100MB for offline reasoning.',
      icon: Zap,
    },
  ]

  const [activeCase, setActiveCase] = useState(0)

  return (
    <section id="usecases" className="section">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="section-title">High-Value Use Cases</h2>
          <p className="section-subtitle mx-auto">
            Real-world applications that demonstrate the power of knowledge compression
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Use Case List */}
          <div className="space-y-4">
            {useCases.map((useCase, index) => (
              <motion.button
                key={useCase.title}
                onClick={() => setActiveCase(index)}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className={`w-full text-left p-6 rounded-xl transition-all duration-300 ${
                  activeCase === index
                    ? 'glass border-primary/50'
                    : 'bg-white/5 hover:bg-white/10'
                }`}
              >
                <div className="flex items-center space-x-4">
                  <useCase.icon className={`w-8 h-8 ${
                    activeCase === index ? 'text-primary' : 'text-gray-500'
                  }`} />
                  <div>
                    <h4 className={`font-semibold ${
                      activeCase === index ? 'text-white' : 'text-gray-400'
                    }`}>
                      {useCase.title}
                    </h4>
                    <p className={`text-sm mt-1 ${
                      activeCase === index ? 'text-gray-300' : 'text-gray-500'
                    }`}>
                      {useCase.desc}
                    </p>
                  </div>
                </div>
              </motion.button>
            ))}
          </div>

          {/* Use Case Visualization */}
          <motion.div
            key={activeCase}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass rounded-2xl p-8 min-h-[300px] flex items-center justify-center"
          >
            <div className="text-center">
              {(() => {
                const ActiveIcon = useCases[activeCase].icon;
                return <ActiveIcon className="w-24 h-24 text-primary mx-auto mb-6" />;
              })()}
              <h3 className="text-2xl font-bold mb-4">{useCases[activeCase].title}</h3>
              <p className="text-gray-300">{useCases[activeCase].desc}</p>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}

// Tech Advantage Section
function TechAdvantage() {
  const advantages = [
    {
      title: 'Compression Ratio',
      desc: 'The higher the ratio with meaning retention, the bigger the moat.',
      icon: TrendingUp,
    },
    {
      title: 'Causal Graph Preservation',
      desc: 'Almost impossible to replicate without deep proprietary tech.',
      icon: Database,
    },
    {
      title: 'Kernel Format Standardization',
      desc: 'Become the PDF for knowledge.',
      icon: FileText,
    },
    {
      title: 'Data Network Effects',
      desc: 'Each kernel improves the global kernel index.',
      icon: Globe,
    },
    {
      title: 'Ecosystem Lock-in',
      desc: 'Agents, apps, AIs all rely on Kernels → hard to switch.',
      icon: Shield,
    },
  ]

  return (
    <section id="tech" className="section bg-surface/50">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="section-title">Tech Advantage / Moat</h2>
          <p className="section-subtitle mx-auto">
            Sustainable competitive advantages that protect our market position
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {advantages.map((advantage, index) => (
            <motion.div
              key={advantage.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
              className="glass rounded-2xl p-6 card-hover"
            >
              <advantage.icon className="w-10 h-10 text-primary mb-4" />
              <h3 className="text-xl font-bold mb-2">{advantage.title}</h3>
              <p className="text-gray-400">{advantage.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

// Revenue Model Section
function Revenue() {
  const revenueStreams = [
    {
      title: 'API Charges',
      desc: 'Per MB of compressed input',
      icon: Zap,
    },
    {
      title: 'Kernel Storage',
      desc: 'Pay to store kernels in the cloud',
      icon: Database,
    },
    {
      title: 'Kernel Query',
      desc: 'Charges per 1,000 queries',
      icon: Search,
    },
    {
      title: 'Kernel Marketplace',
      desc: 'Sell domain kernels to enterprises',
      icon: Globe,
    },
    {
      title: 'Enterprise Plans',
      desc: 'Custom deployments and SLAs',
      icon: Building2,
    },
    {
      title: 'Fine-Tuning Kits',
      desc: 'LLMs fine-tuned on kernels only',
      icon: Brain,
    },
  ]

  return (
    <section id="revenue" className="section">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="section-title">Revenue Model</h2>
          <p className="section-subtitle mx-auto">
            Multiple monetization streams for sustainable growth
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {revenueStreams.map((stream, index) => (
            <motion.div
              key={stream.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
              className="glass rounded-2xl p-6 card-hover"
            >
              <stream.icon className="w-10 h-10 text-primary mb-4" />
              <h3 className="text-xl font-bold mb-2">{stream.title}</h3>
              <p className="text-gray-400">{stream.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

// Market Size Section
function Market() {
  return (
    <section id="market" className="section bg-surface/50">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="section-title">Market Size</h2>
          <p className="section-subtitle mx-auto">
            Massive total addressable market across multiple verticals
          </p>
        </motion.div>

        <div className="glass rounded-3xl p-12 text-center mb-12">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            whileInView={{ scale: 1, opacity: 1 }}
            viewport={{ once: true }}
            className="mb-8"
          >
            <div className="text-6xl md:text-8xl font-bold gradient-text mb-4">$3T+</div>
            <div className="text-xl text-gray-400">Combined TAM</div>
          </motion.div>

          <p className="text-gray-300 text-lg max-w-3xl mx-auto mb-12">
            This product touches AI infrastructure, knowledge management, enterprise search, 
            LLM optimization, education, healthcare, and government.
          </p>

          <div className="text-4xl md:text-5xl font-bold gradient-text mb-4">
            If Kernelize becomes the standard knowledge format → $10T empire potential
          </div>
        </div>

        {/* Market Verticals */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            'AI Infrastructure',
            'Knowledge Management',
            'Enterprise Search',
            'LLM Optimization',
            'Education',
            'Healthcare',
            'Landmark',
            'Research',
          ].map((vertical, index) => (
            <motion.div
              key={vertical}
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="glass rounded-xl p-4 text-center"
            >
              <CheckCircle className="w-6 h-6 text-primary mx-auto mb-2" />
              <span className="text-sm font-medium">{vertical}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

// GTM Strategy Section
function GTM() {
  const phases = [
    {
      phase: 'Phase 1',
      title: 'Developer Adoption',
      desc: 'Release open-source Kernel SDK',
      icon: Code,
    },
    {
      phase: 'Phase 2',
      title: 'Enterprise',
      desc: 'Solve internal knowledge search, agent knowledge packs, and LLM token reduction',
      icon: Building2,
    },
    {
      phase: 'Phase 3',
      title: 'Platform',
      desc: 'Launch Kernel Marketplace for domain-specific kernels',
      icon: Globe,
    },
    {
      phase: 'Phase 4',
      title: 'Global Standard',
      desc: 'Become the semantic compression layer of the planet',
      icon: Rocket,
    },
  ]

  return (
    <section id="gtm" className="section">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="section-title">Go-To-Market Strategy</h2>
          <p className="section-subtitle mx-auto">
            A phased approach to capture the market and establish dominance
          </p>
        </motion.div>

        <div className="relative">
          {/* Connection Line */}
          <div className="absolute top-12 left-0 right-0 h-0.5 bg-gradient-to-r from-primary via-secondary to-accent hidden md:block" />

          <div className="grid md:grid-cols-4 gap-8">
            {phases.map((phase, index) => (
              <motion.div
                key={phase.phase}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                className="relative"
              >
                <div className="glass rounded-2xl p-6 text-center h-full">
                  <div className="w-12 h-12 bg-gradient-to-br from-primary to-secondary rounded-full flex items-center justify-center mx-auto mb-4 relative z-10">
                    <phase.icon className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-primary text-sm font-semibold mb-2">{phase.phase}</div>
                  <h3 className="text-xl font-bold mb-2">{phase.title}</h3>
                  <p className="text-gray-400 text-sm">{phase.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

// CTA Section
function CTA() {
  return (
    <section className="section relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0">
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{ duration: 8, repeat: Infinity }}
          className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/20 rounded-full blur-3xl"
        />
      </div>

      <div className="max-w-4xl mx-auto text-center relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Ready to Compress the World's Knowledge?
          </h2>
          <p className="text-xl text-gray-400 mb-10">
            Join the revolution in semantic knowledge compression
          </p>

          {/* Pitch Sentence */}
          <div className="glass rounded-2xl p-8 mb-10">
            <p className="text-2xl md:text-3xl font-bold gradient-text leading-relaxed">
              "KERNELIZE compresses the world's knowledge into ultra-dense intelligence kernels 
              that AI systems can search, reason over, and learn from – 100× faster, 100× cheaper, 
              with near-perfect meaning preservation."
            </p>
          </div>

          {/* Email Capture */}
          <div className="flex flex-col sm:flex-row gap-4 max-w-xl mx-auto">
            <input
              type="email"
              placeholder="Enter your email"
              className="flex-1 px-6 py-4 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-primary"
            />
            <button className="btn-primary whitespace-nowrap">
              Get Early Access
            </button>
          </div>

          <p className="text-gray-500 text-sm mt-6">
            Join 10,000+ developers and enterprises waiting for the revolution
          </p>
        </motion.div>
      </div>
    </section>
  )
}

// Footer
function Footer() {
  return (
    <footer className="border-t border-white/10 py-12">
      <div className="max-w-6xl mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-8 mb-8">
          <div>
            <a href="#" className="flex items-center space-x-2 mb-4">
              <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center">
                <Cpu className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold font-display">KERNELIZE</span>
            </a>
            <p className="text-gray-400 text-sm">
              Compress the world's knowledge into ultra-dense intelligence kernels.
            </p>
          </div>
          
          {[
            {
              title: 'Product',
              links: ['Kernel Engine', 'API Documentation', 'SDK', 'Pricing'],
            },
            {
              title: 'Company',
              links: ['About', 'Blog', 'Careers', 'Contact'],
            },
            {
              title: 'Legal',
              links: ['Privacy', 'Terms', 'Security'],
            },
          ].map((section) => (
            <div key={section.title}>
              <h4 className="font-semibold mb-4">{section.title}</h4>
              <ul className="space-y-2">
                {section.links.map((link) => (
                  <li key={link}>
                    <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors">
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row items-center justify-between">
          <p className="text-gray-500 text-sm">
            © 2024 KERNELIZE. All rights reserved.
          </p>
          <div className="flex items-center space-x-4 mt-4 md:mt-0">
            <a href="#" className="text-gray-400 hover:text-white transition-colors">
              Twitter
            </a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">
              GitHub
            </a>
            <a href="#" className="text-gray-400 hover:text-white transition-colors">
              LinkedIn
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}

// Main Page Component
export default function Home() {
  return (
    <main className="min-h-screen animated-bg">
      <Navigation />
      <Hero />
      <Vision />
      <Product />
      <Components />
      <Customers />
      <UseCases />
      <TechAdvantage />
      <Revenue />
      <Market />
      <GTM />
      <CTA />
      <Footer />
    </main>
  )
}

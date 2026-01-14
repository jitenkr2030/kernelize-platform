/**
 * KERNELIZE Platform - Landing Page
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

'use client'

import { motion } from 'framer-motion'
import { 
  Zap, 
  ArrowUpRight, 
  Shield, 
  Globe, 
  Database, 
  Cloud,
  Server,
  Users,
  Lock,
  Sparkles,
  Cpu,
  Layers,
  BarChart3,
  Terminal,
  ShieldCheck,
  Globe2,
  Rocket
} from 'lucide-react'
import { Card } from '@/components/ui/Card'

// Bento grid feature data
const BentoFeatures = [
  {
    title: 'AI-Powered Compression',
    description: 'Reduce storage costs by up to 90% with our proprietary lossless AI compression technology.',
    icon: Sparkles,
    gradient: 'from-purple-500 to-pink-500',
    size: 'lg',
    stats: ['90% Storage Reduction', 'Zero Data Loss'],
  },
  {
    title: 'Real-time Analytics',
    description: 'Process millions of events per second with sub-millisecond latency.',
    icon: BarChart3,
    gradient: 'from-blue-500 to-cyan-500',
    size: 'md',
    stats: ['1M+ Events/sec', '<1ms Latency'],
  },
  {
    title: 'Enterprise Security',
    description: 'Bank-grade encryption with SOC2, HIPAA, and ISO 27001 compliance.',
    icon: ShieldCheck,
    gradient: 'from-green-500 to-emerald-500',
    size: 'sm',
    stats: ['SOC2 Type II', '256-bit AES'],
  },
  {
    title: 'Global CDN',
    description: 'Deploy to 300+ edge locations worldwide for optimal performance.',
    icon: Globe2,
    gradient: 'from-orange-500 to-amber-500',
    size: 'sm',
    stats: ['300+ Edges', '<50ms Global'],
  },
  {
    title: 'Auto Scaling',
    description: 'Automatically scale from zero to millions of requests without configuration.',
    icon: ArrowUpRight,
    gradient: 'from-indigo-500 to-purple-500',
    size: 'md',
    stats: ['Zero to Millions', 'Instant Scale'],
  },
  {
    title: 'Infrastructure as Code',
    description: 'Define your entire infrastructure with simple YAML or Terraform configurations.',
    icon: Terminal,
    gradient: 'from-cyan-500 to-blue-500',
    size: 'lg',
    stats: ['Terraform Support', 'GitOps Ready'],
  },
  {
    title: 'Multi-Cloud Support',
    description: 'Deploy seamlessly across AWS, GCP, Azure, or on-premise environments.',
    icon: Cloud,
    gradient: 'from-pink-500 to-rose-500',
    size: 'md',
    stats: ['3 Cloud Providers', 'Hybrid Support'],
  },
  {
    title: '24/7 Expert Support',
    description: 'Dedicated support team available around the clock for critical issues.',
    icon: Users,
    gradient: 'from-violet-500 to-purple-500',
    size: 'sm',
    stats: ['24/7 Available', '<15min Response'],
  },
]

export default function Features() {
  return (
    <section id="features" className="section-padding relative">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-b from-primary/3 via-transparent to-transparent" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent" />
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent" />

      <div className="container-custom relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full mb-6">
            <Rocket className="w-4 h-4 text-accent" />
            <span className="text-sm text-text-secondary">Enterprise-Grade Capabilities</span>
          </div>
          <h2 className="section-title">
            Everything you need to build
            <span className="block gradient-text mt-2">the future of AI infrastructure</span>
          </h2>
          <p className="section-subtitle mx-auto">
            From compression to deployment, our comprehensive platform provides 
            all the tools your team needs to succeed at scale.
          </p>
        </motion.div>

        {/* Bento Grid Layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {BentoFeatures.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className={`${
                feature.size === 'lg' 
                  ? 'lg:col-span-2 lg:row-span-2' 
                  : feature.size === 'md'
                  ? 'lg:col-span-2'
                  : ''
              }`}
            >
              <Card 
                variant="bordered" 
                hover 
                className={`h-full group relative overflow-hidden ${
                  feature.size === 'lg' ? 'p-8' : 'p-6'
                }`}
              >
                {/* Gradient background glow */}
                <div className={`absolute top-0 right-0 w-32 h-32 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-10 rounded-full blur-3xl transition-opacity duration-500`} />
                
                {/* Icon */}
                <div className={`${feature.size === 'lg' ? 'w-16 h-16' : 'w-12 h-12'} rounded-2xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className={`${feature.size === 'lg' ? 'w-8 h-8' : 'w-6 h-6'} text-white`} />
                </div>

                {/* Content */}
                <h3 className={`${feature.size === 'lg' ? 'text-2xl' : 'text-xl'} font-semibold text-text-primary mb-3 group-hover:text-white transition-colors`}>
                  {feature.title}
                </h3>
                <p className="text-text-secondary mb-6 leading-relaxed">
                  {feature.description}
                </p>

                {/* Stats */}
                <div className="flex flex-wrap gap-2">
                  {feature.stats.map((stat, statIndex) => (
                    <span
                      key={statIndex}
                      className={`px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${feature.gradient} bg-opacity-10 text-white border border-white/10`}
                    >
                      {stat}
                    </span>
                  ))}
                </div>

                {/* Hover arrow indicator */}
                <div className="absolute bottom-6 right-6 w-10 h-10 rounded-full bg-white/5 flex items-center justify-center opacity-0 group-hover:opacity-100 group-hover:translate-x-0 -translate-x-2 transition-all duration-300">
                  <ArrowUpRight className="w-5 h-5 text-white" />
                </div>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Integration section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-20 text-center"
        >
          <p className="text-text-secondary mb-8 flex items-center justify-center gap-2">
            <Layers className="w-5 h-5" />
            Seamlessly integrates with your existing stack
          </p>
          <div className="flex flex-wrap items-center justify-center gap-8 opacity-60">
            {[
              { name: 'AWS', color: '#FF9900' },
              { name: 'Google Cloud', color: '#4285F4' },
              { name: 'Azure', color: '#0078D4' },
              { name: 'Kubernetes', color: '#326CE5' },
              { name: 'Docker', color: '#2496ED' },
              { name: 'Terraform', color: '#7B42BC' },
              { name: 'PostgreSQL', color: '#4169E1' },
              { name: 'Redis', color: '#DC382D' },
            ].map((company) => (
              <motion.div
                key={company.name}
                className="flex items-center gap-2 px-4 py-2 glass rounded-lg hover:bg-white/5 transition-colors cursor-pointer"
                whileHover={{ scale: 1.05 }}
              >
                <div 
                  className="w-6 h-6 rounded flex items-center justify-center text-white text-xs font-bold"
                  style={{ backgroundColor: company.color }}
                >
                  {company.name.charAt(0)}
                </div>
                <span className="text-sm font-medium text-text-secondary">{company.name}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Code snippet showcase */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-20"
        >
          <div className="glass rounded-2xl overflow-hidden">
            <div className="bg-surfaceLight/80 px-6 py-4 flex items-center gap-3 border-b border-white/5">
              <div className="flex gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
              </div>
              <span className="text-text-muted text-sm font-mono">kernelize.yaml</span>
            </div>
            <div className="p-6 overflow-x-auto">
              <pre className="font-mono text-sm">
                <code>
                  <span className="text-purple-400">name</span>: <span className="text-green-400">my-ai-service</span>
                  {'\n'}
                  <span className="text-purple-400">version</span>: <span className="text-blue-400">2.0.0</span>
                  {'\n'}
                  <span className="text-purple-400">services</span>:
                  {'\n'}
                  {'  '}<span className="text-purple-400">api</span>:
                  {'\n'}
                  {'    '}<span className="text-purple-400">image</span>: <span className="text-green-400">kernelize/ai-inference:v2</span>
                  {'\n'}
                  {'    '}<span className="text-purple-400">replicas</span>: <span className="text-blue-400">auto</span>
                  {'\n'}
                  {'    '}<span className="text-purple-400">resources</span>:
                  {'\n'}
                  {'      '}<span className="text-purple-400">cpu</span>: <span className="text-blue-400">4</span>
                  {'\n'}
                  {'      '}<span className="text-purple-400">memory</span>: <span className="text-blue-400">16Gi</span>
                  {'\n'}
                  {'    '}<span className="text-purple-400">autoscale</span>:
                  {'\n'}
                  {'      '}<span className="text-purple-400">min</span>: <span className="text-blue-400">1</span>
                  {'\n'}
                  {'      '}<span className="text-purple-400">max</span>: <span className="text-blue-400">100</span>
                  {'\n'}
                  {'      '}<span className="text-purple-400">metrics</span>: <span className="text-green-400">[cpu, latency]</span>
                  {'\n'}
                  <span className="text-purple-400">features</span>:
                  {'\n'}
                  {'  '}<span className="text-purple-400">- compression</span>: <span className="text-green-400">ai-enhanced</span>
                  {'\n'}
                  {'  '}<span className="text-purple-400">- monitoring</span>: <span className="text-green-400">enabled</span>
                  {'\n'}
                  {'  '}<span className="text-purple-400">- security</span>: <span className="text-green-400">enterprise</span>
                </code>
              </pre>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

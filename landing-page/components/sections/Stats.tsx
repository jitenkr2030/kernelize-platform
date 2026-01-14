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
import { Server, Globe2, Users, Zap, Shield, Database, Cloud, TrendingUp } from 'lucide-react'

const statsData = [
  {
    icon: Server,
    value: '99.99',
    suffix: '%',
    label: 'Uptime SLA',
    description: 'Guaranteed availability with redundant infrastructure',
    gradient: 'from-green-500 to-emerald-500',
    trend: '+0.01%',
  },
  {
    icon: Database,
    value: '50',
    suffix: 'PB+',
    label: 'Data Processed',
    description: 'Petabytes of data processed monthly',
    gradient: 'from-blue-500 to-cyan-500',
    trend: '+12%',
  },
  {
    icon: Globe2,
    value: '300',
    suffix: '+',
    label: 'Edge Locations',
    description: 'Global edge servers for low latency',
    gradient: 'from-purple-500 to-pink-500',
    trend: '+50',
  },
  {
    icon: Users,
    value: '10K',
    suffix: '+',
    label: 'API Calls/Sec',
    description: 'Requests handled per second',
    gradient: 'from-orange-500 to-amber-500',
    trend: '+25%',
  },
  {
    icon: Shield,
    value: '500',
    suffix: '+',
    label: 'Enterprise Users',
    description: 'Companies trusting our platform',
    gradient: 'from-indigo-500 to-violet-500',
    trend: '+100',
  },
  {
    icon: Zap,
    value: '<1',
    suffix: 'ms',
    label: 'Average Latency',
    description: 'Sub-millisecond response times',
    gradient: 'from-cyan-500 to-blue-500',
    trend: '-0.2ms',
  },
  {
    icon: TrendingUp,
    value: '10X',
    suffix: '',
    label: 'Faster Deployments',
    description: 'Compared to traditional methods',
    gradient: 'from-rose-500 to-pink-500',
    trend: 'New',
  },
  {
    icon: Cloud,
    value: '90',
    suffix: '%',
    label: 'Cost Reduction',
    description: 'Average savings on infrastructure',
    gradient: 'from-teal-500 to-cyan-500',
    trend: 'Avg',
  },
]

export default function Stats() {
  return (
    <section className="py-20 border-y border-white/5 relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-transparent to-accent/5" />
      
      <div className="container-custom relative z-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {statsData.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="group"
            >
              <div className="glass rounded-2xl p-6 h-full hover-border-gradient transition-all duration-500">
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${stat.gradient} flex items-center justify-center opacity-80 group-hover:opacity-100 transition-opacity`}>
                    <stat.icon className="w-6 h-6 text-white" />
                  </div>
                  <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                    stat.trend.startsWith('+') || stat.trend === 'New' 
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-accent/20 text-accent'
                  }`}>
                    {stat.trend}
                  </span>
                </div>
                
                {/* Value */}
                <div className="flex items-baseline gap-1 mb-2">
                  <span className="text-4xl font-bold gradient-text">{stat.value}</span>
                  <span className="text-lg text-text-secondary">{stat.suffix}</span>
                </div>
                
                {/* Label & Description */}
                <div>
                  <p className="text-text-primary font-semibold mb-1">{stat.label}</p>
                  <p className="text-text-muted text-sm">{stat.description}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

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
import { FEATURES } from '@/lib/constants'
import { Card } from '@/components/ui/Card'

export default function Features() {
  return (
    <section id="features" className="section-padding relative">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-b from-primary/3 via-transparent to-transparent" />

      <div className="container-custom relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <h2 className="section-title">
            Enterprise-Grade Features for
            <span className="gradient-text"> Modern Data Teams</span>
          </h2>
          <p className="section-subtitle">
            Everything you need to build, deploy, and scale your data infrastructure 
            with enterprise security and compliance built-in.
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {FEATURES.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card variant="bordered" hover className="h-full group">
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-text-primary mb-2">
                  {feature.title}
                </h3>
                <p className="text-text-secondary">
                  {feature.description}
                </p>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Integration Logos */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="mt-20 text-center"
        >
          <p className="text-text-secondary mb-8">Trusted by leading enterprises worldwide</p>
          <div className="flex flex-wrap items-center justify-center gap-12 opacity-50">
            {[
              { name: 'AWS', color: '#FF9900' },
              { name: 'Azure', color: '#0078D4' },
              { name: 'Google Cloud', color: '#4285F4' },
              { name: 'Snowflake', color: '#29B5E8' },
              { name: 'Databricks', color: '#FF3621' },
              { name: 'Kubernetes', color: '#326CE5' },
            ].map((company) => (
              <div
                key={company.name}
                className="text-2xl font-bold text-slate-400 hover:text-text-primary transition-colors cursor-pointer"
                style={{ color: company.color }}
              >
                {company.name}
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}

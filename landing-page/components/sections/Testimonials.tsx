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
import { Star, Quote, ArrowRight, Building2, Code2, Database, Globe, Shield } from 'lucide-react'

const testimonials = [
  {
    content: "KERNELIZE has completely transformed how we deploy and scale our AI applications. We've reduced our deployment time by 80% and our infrastructure costs by 60%. The AI compression feature alone saves us terabytes of storage every month.",
    author: "Sarah Chen",
    role: "CTO",
    company: "TechFlow Inc.",
    avatar: "SC",
    rating: 5,
    companyType: "Technology",
    gradient: "from-blue-500 to-cyan-500",
  },
  {
    content: "The enterprise security features gave us the confidence to migrate our entire infrastructure to the cloud. SOC2 compliance was a breeze, and their 24/7 support team is exceptional. Best decision we made for our platform.",
    author: "Michael Rodriguez",
    role: "VP of Engineering",
    company: "SecureNet Solutions",
    avatar: "MR",
    rating: 5,
    companyType: "Security",
    gradient: "from-green-500 to-emerald-500",
  },
  {
    content: "Outstanding support team and seamless scaling capabilities. KERNELIZE is now essential to our business. We went from 10K to 10M requests per day without any downtime or performance issues. Absolutely remarkable platform.",
    author: "Emily Watson",
    role: "Lead Developer",
    company: "CloudFirst Labs",
    avatar: "EW",
    rating: 5,
    companyType: "Cloud Services",
    gradient: "from-purple-500 to-pink-500",
  },
  {
    content: "As a fintech startup, we needed enterprise-grade infrastructure with strict compliance requirements. KERNELIZE delivered beyond our expectations. The documentation is excellent and the API is a joy to work with.",
    author: "David Park",
    role: "Founder & CEO",
    company: "FinanceHub",
    avatar: "DP",
    rating: 5,
    companyType: "Fintech",
    gradient: "from-orange-500 to-amber-500",
  },
  {
    content: "The real-time analytics capabilities are game-changing. We can now process millions of events per second with sub-millisecond latency. Our data team is more productive than ever before.",
    author: "Lisa Thompson",
    role: "Head of Data",
    company: "DataDriven Co.",
    avatar: "LT",
    rating: 5,
    companyType: "Data Analytics",
    gradient: "from-indigo-500 to-violet-500",
  },
  {
    content: "Migrating from our legacy system was seamless. KERNELIZE's Terraform support made it easy to define our entire infrastructure as code. The team was productive from day one.",
    author: "James Wilson",
    role: "DevOps Lead",
    company: "ScaleUp Ventures",
    avatar: "JW",
    rating: 5,
    companyType: "Venture Capital",
    gradient: "from-rose-500 to-pink-500",
  },
]

const companyLogos = [
  { name: 'TechFlow', icon: Code2, color: '#3B82F6' },
  { name: 'SecureNet', icon: Shield, color: '#10B981' },
  { name: 'CloudFirst', icon: Globe, color: '#8B5CF6' },
  { name: 'FinanceHub', icon: Building2, color: '#F59E0B' },
  { name: 'DataDriven', icon: Database, color: '#EC4899' },
  { name: 'ScaleUp', icon: ArrowRight, color: '#6366F1' },
]

export default function Testimonials() {
  return (
    <section id="testimonials" className="section-padding relative">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-b from-primary/2 via-transparent to-transparent" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent" />

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
            <Quote className="w-4 h-4 text-accent" />
            <span className="text-sm text-text-secondary">Trusted by Industry Leaders</span>
          </div>
          <h2 className="section-title">
            Loved by engineering teams
            <span className="block gradient-text mt-2">at world-class companies</span>
          </h2>
          <p className="section-subtitle mx-auto">
            See what our customers are saying about their experience 
            building with the KERNELIZE Platform.
          </p>
        </motion.div>

        {/* Company logos strip */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="flex items-center justify-center gap-8 mb-16 overflow-hidden"
        >
          <div className="flex items-center gap-8 animate-marquee whitespace-nowrap">
            {[...companyLogos, ...companyLogos, ...companyLogos].map((company, index) => (
              <div
                key={`${company.name}-${index}`}
                className="flex items-center gap-2 px-4 py-2 glass rounded-lg"
              >
                <div 
                  className="w-8 h-8 rounded flex items-center justify-center text-white text-xs font-bold"
                  style={{ backgroundColor: company.color }}
                >
                  <company.icon className="w-4 h-4" />
                </div>
                <span className="text-sm font-medium text-text-secondary">{company.name}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Testimonials Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={testimonial.author}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
            >
              <div className="glass rounded-2xl p-8 h-full hover-border-gradient group transition-all duration-500">
                {/* Company badge */}
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-2">
                    <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${testimonial.gradient} flex items-center justify-center text-white text-xs font-bold`}>
                      {testimonial.avatar}
                    </div>
                    <div>
                      <div className="font-semibold text-text-primary text-sm">{testimonial.author}</div>
                      <div className="text-text-muted text-xs">{testimonial.role}</div>
                    </div>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium bg-gradient-to-r ${testimonial.gradient} bg-opacity-10 border border-white/10`}>
                    {testimonial.companyType}
                  </span>
                </div>

                {/* Rating */}
                <div className="flex items-center gap-1 mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 text-yellow-500 fill-yellow-500" />
                  ))}
                </div>

                {/* Content */}
                <p className="text-text-secondary leading-relaxed mb-6 text-sm">
                  "{testimonial.content}"
                </p>

                {/* Company */}
                <div className="flex items-center gap-2 pt-4 border-t border-white/5">
                  <Building2 className="w-4 h-4 text-text-muted" />
                  <span className="text-text-muted text-sm">{testimonial.company}</span>
                </div>

                {/* Hover quote icon */}
                <div className="absolute top-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Quote className="w-8 h-8 text-primary/20" />
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="mt-16 text-center"
        >
          <div className="glass rounded-2xl p-8 max-w-2xl mx-auto">
            <p className="text-text-secondary mb-4">
              Ready to join these industry leaders and transform your infrastructure?
            </p>
            <a
              href="/signup"
              className="inline-flex items-center gap-2 text-primary hover:text-primary-light font-medium transition-colors group"
            >
              Start your free trial today
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </a>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

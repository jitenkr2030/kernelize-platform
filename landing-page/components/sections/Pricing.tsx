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
import { Check } from 'lucide-react'
import { PRICING_PLANS } from '@/lib/constants'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import Link from 'next/link'

export default function Pricing() {
  return (
    <section id="pricing" className="section-padding relative">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-b from-accent/3 via-transparent to-transparent" />

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
            Simple, Transparent
            <span className="gradient-text"> Pricing</span>
          </h2>
          <p className="section-subtitle">
            Choose the plan that fits your needs. All plans include our 
            industry-leading AI compression technology.
          </p>
        </motion.div>

        {/* Pricing Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {PRICING_PLANS.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`relative ${plan.popular ? 'lg:-mt-4' : ''}`}
            >
              {/* Popular Badge */}
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                  <span className="bg-gradient-to-r from-primary to-accent text-white text-sm font-medium px-4 py-1 rounded-full">
                    Most Popular
                  </span>
                </div>
              )}

              <Card
                variant={plan.popular ? 'default' : 'bordered'}
                className={`h-full flex flex-col ${
                  plan.popular
                    ? 'border-2 border-primary shadow-lg shadow-primary/20'
                    : ''
                }`}
              >
                {/* Plan Header */}
                <div className="mb-6">
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${plan.gradient} flex items-center justify-center mb-4`}>
                    <span className="text-white font-bold text-lg">
                      {plan.name.charAt(0)}
                    </span>
                  </div>
                  <h3 className="text-xl font-semibold text-text-primary mb-1">
                    {plan.name}
                  </h3>
                  <p className="text-text-secondary text-sm">
                    {plan.description}
                  </p>
                </div>

                {/* Price */}
                <div className="mb-6">
                  {plan.price !== null ? (
                    <div className="flex items-baseline">
                      <span className="text-4xl font-bold text-text-primary">
                        ${plan.price}
                      </span>
                      <span className="text-text-secondary ml-1">/month</span>
                    </div>
                  ) : (
                    <div className="text-2xl font-bold text-text-primary">
                      Custom Pricing
                    </div>
                  )}
                </div>

                {/* Features List */}
                <ul className="space-y-3 mb-8 flex-grow">
                  {plan.features.map((feature) => (
                    <li key={feature} className="flex items-start gap-3">
                      <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                      <span className="text-text-secondary text-sm">
                        {feature}
                      </span>
                    </li>
                  ))}
                </ul>

                {/* CTA Button */}
                <Link href={plan.price !== null ? "/signup" : "#contact"}>
                  <Button
                    variant={plan.popular ? 'primary' : 'secondary'}
                    className="w-full"
                  >
                    {plan.cta}
                  </Button>
                </Link>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Money Back Guarantee */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="mt-16 text-center"
        >
          <div className="glass rounded-2xl p-8 max-w-2xl mx-auto">
            <h3 className="text-lg font-semibold text-text-primary mb-2">
              30-Day Money-Back Guarantee
            </h3>
            <p className="text-text-secondary">
              Try KERNELIZE risk-free. If you're not completely satisfied with our 
              platform within the first 30 days, we'll refund your payment, no questions asked.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

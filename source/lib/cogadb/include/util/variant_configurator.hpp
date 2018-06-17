/*
 * File:   variant_configurator.hpp
 * Author: christoph
 *
 * Created on 25. May 2016, 15:49
 */

#ifndef VARIANT_CONFIGURATOR_HPP
#define VARIANT_CONFIGURATOR_HPP

#include <core/variable_manager.hpp>
#include <parser/client.hpp>
#include <util/opencl_runtime.hpp>

#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace CoGaDB {

  typedef std::map<std::string, std::string> Variant;

  void print(ClientPtr client, const Variant& variant);

  /**
   * @brief An iterator like class that iterates over multiple
   * vectors and returns a tuple (i.e. permutation) of their values.
   */
  class VariantIterator {
   private:
    /**
     * @brief Tells wether the iterator is at the first element
     * (true) or exhausted (false).
     */
    bool first = true;

   public:
    typedef std::vector<Variant> VariantVector;
    typedef std::vector<std::pair<std::string, std::vector<std::string>>>
        FlattenedDimensions;

    /**
     * @brief Represents a dimension from which one of the elements
     * of the Variant should be drawn.
     */
    class Dimension {
     public:
      /**
       * @brief Construct an instance of Dimension.
       * @param _values Values of that dimension.
       */
      Dimension(const std::string& name, const std::vector<std::string>& values,
                uint32_t priority);

      void addChilds(const std::vector<std::string>& values,
                     const std::string& child_dimension,
                     const std::vector<std::string>& child_options);

      VariantVector::iterator begin();
      VariantVector::iterator end();

      std::vector<std::string> getAllValues();

      FlattenedDimensions getFlattenedDimensions();

      const std::string& getName() const { return name_; }

      bool operator<(const Dimension& other) const {
        return priority_ < other.priority_;
      }

      bool operator>(const Dimension& other) const { return other < *this; }

     private:
      void generateVariants();

      typedef std::map<std::string, VariantIterator> ValueMap;
      /**
       * @brief Values of that dimension
       */
      ValueMap values_;
      std::string name_;
      uint32_t priority_;

      VariantVector variants_;
      bool variants_need_refresh_ = true;
    };

    typedef std::map<std::string, Dimension> DimensionMap;
    typedef std::vector<Dimension> DimensionVector;

    /**
     * @brief Construct an instance of VariantIterator.
     */
    VariantIterator() {}

    /**
     * @brief Add another dimenion to the iterator and reset state.
     * @param dimension Name of the dimension
     * @param options Values of that dimension
     */
    Dimension& add(const std::string& dimension,
                   const std::vector<std::string>& options,
                   uint32_t priority = 0);

    VariantVector::iterator begin();
    VariantVector::iterator end();
    bool empty() const { return pool_.empty(); }

    DimensionVector& getSortedDimensions();

    FlattenedDimensions getFlattenedDimensions();

   private:
    void refreshVariants();
    VariantVector generateAllVariants(DimensionMap::iterator next_dim);

    /**
     * @brief Map of dimensions from which the permutation should be
     * drawn.
     */
    DimensionMap pool_;
    VariantVector variants_;
    DimensionVector pool_sorted_;
    bool variants_need_refresh_ = true;
    bool pool_sorted_need_refresh = true;
  };

  /**
   * @brief A configurator for VariableManager
   */
  class VariantConfigurator {
   public:
    /**
     * @brief Set the variables in VariableManager according to the
     * keys and their values in the Variant.
     * @param variant Map of keys and values (Variant)
     */
    void operator()(const Variant& variant);
  };

  /**
   * @brief Blacklist of dimension values
   */
  class VariantBlacklist {
   private:
    std::map<std::string, std::map<std::string, bool>> values;

   public:
    VariantBlacklist(){};
    VariantBlacklist(std::map<std::string, std::map<std::string, bool>> values);
    void add(const std::string& dimension, const std::string& value);
    bool contains(const std::string& dimension, const std::string& value);
    bool is_blacklisted(const Variant& variant);
  };

}  // end namespace CogaDB

#endif /* VARIANT_CONFIGURATOR_HPP */

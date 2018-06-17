#include <util/variant_configurator.hpp>

#include <iostream>
#include <thread>

namespace CoGaDB {

void print(ClientPtr client, const Variant& variant) {
  std::ostream& out = client->getOutputStream();
  Variant::const_iterator cit;
  std::cout << "Variant: " << std::endl;
  for (cit = variant.begin(); cit != variant.end(); ++cit) {
    out << "\t" << cit->first << " = " << cit->second << std::endl;
  }
}

VariantIterator::Dimension::Dimension(const std::string& name,
                                      const std::vector<std::string>& values,
                                      uint32_t priority)
    : name_(name), priority_(priority) {
  for (const auto& value : values) {
    values_.insert(std::make_pair(value, VariantIterator()));
  }
}

void VariantIterator::Dimension::addChilds(
    const std::vector<std::string>& values, const std::string& child_dimension,
    const std::vector<std::string>& child_options) {
  for (const auto& value : values) {
    values_[value].add(child_dimension, child_options);
  }

  variants_need_refresh_ = true;
}

VariantIterator::VariantVector::iterator VariantIterator::Dimension::begin() {
  if (variants_need_refresh_) {
    generateVariants();
  }

  return variants_.begin();
}

VariantIterator::VariantVector::iterator VariantIterator::Dimension::end() {
  if (variants_need_refresh_) {
    generateVariants();
  }

  return variants_.end();
}

void VariantIterator::Dimension::generateVariants() {
  variants_need_refresh_ = false;
  variants_.clear();

  for (auto& option : values_) {
    if (option.second.empty()) {
      Variant variant;
      variant.insert(std::make_pair(name_, option.first));
      variants_.push_back(variant);
    } else {
      for (const auto& child_variant : option.second) {
        Variant variant;

        variant.insert(std::make_pair(name_, option.first));
        variant.insert(child_variant.begin(), child_variant.end());
        variants_.push_back(variant);
      }
    }
  }
}

VariantIterator::Dimension& VariantIterator::add(
    const std::string& dimension, const std::vector<std::string>& options,
    uint32_t priority) {
  auto insert = pool_.insert(
      std::make_pair(dimension, Dimension(dimension, options, priority)));
  variants_need_refresh_ = true;
  pool_sorted_need_refresh = true;

  return insert.first->second;
}

VariantIterator::VariantVector::iterator VariantIterator::begin() {
  if (variants_need_refresh_) {
    refreshVariants();
  }

  return variants_.begin();
}

VariantIterator::VariantVector::iterator VariantIterator::end() {
  if (variants_need_refresh_) {
    refreshVariants();
  }

  return variants_.end();
}

void VariantIterator::refreshVariants() {
  variants_need_refresh_ = false;
  variants_.clear();
  variants_ = generateAllVariants(pool_.begin());
}

VariantIterator::VariantVector VariantIterator::generateAllVariants(
    DimensionMap::iterator next_dim) {
  if (next_dim == pool_.end()) {
    return VariantVector();
  }

  VariantVector result;

  auto next_variants = generateAllVariants(std::next(next_dim, 1));

  for (const auto& variant : next_dim->second) {
    if (next_variants.empty()) {
      result.push_back(variant);
    } else {
      for (const auto& next_variant : next_variants) {
        auto variant_new = variant;
        variant_new.insert(next_variant.begin(), next_variant.end());

        result.push_back(variant_new);
      }
    }
  }

  return result;
}

VariantIterator::DimensionVector& VariantIterator::getSortedDimensions() {
  if (pool_sorted_need_refresh) {
    pool_sorted_.clear();
    for (const auto& dim : pool_) {
      pool_sorted_.push_back(dim.second);
    }
    std::sort(pool_sorted_.begin(), pool_sorted_.end(),
              std::greater<Dimension>());
    pool_sorted_need_refresh = false;
  }

  return pool_sorted_;
}

std::vector<std::string> VariantIterator::Dimension::getAllValues() {
  std::vector<std::string> result;

  for (const auto& itr : values_) {
    result.push_back(itr.first);
  }

  return result;
}

void mergeFlattened(const VariantIterator::FlattenedDimensions& insert,
                    VariantIterator::FlattenedDimensions& result) {
  for (auto& dim : insert) {
    result.push_back(std::make_pair(dim.first, dim.second));
  }
}

VariantIterator::FlattenedDimensions
VariantIterator::Dimension::getFlattenedDimensions() {
  FlattenedDimensions result;

  for (auto& itr : values_) {
    auto flattened = itr.second.getFlattenedDimensions();
    mergeFlattened(flattened, result);
  }

  return result;
}

VariantIterator::FlattenedDimensions VariantIterator::getFlattenedDimensions() {
  auto& sorted = getSortedDimensions();
  FlattenedDimensions result;

  for (auto& dim : sorted) {
    auto dim_values = dim.getAllValues();
    result.push_back(std::make_pair(dim.getName(), dim_values));

    auto flattened = dim.getFlattenedDimensions();
    mergeFlattened(flattened, result);
  }

  return result;
}

VariantBlacklist::VariantBlacklist(
    std::map<std::string, std::map<std::string, bool>> values)
    : values(values) {}

void VariantBlacklist::add(const std::string& dimension,
                           const std::string& value) {
  // Find dimension
  auto dim_entry = values.find(dimension);
  if (dim_entry == values.end()) {
    values.insert({dimension, std::map<std::string, bool>()});
    dim_entry = values.find(dimension);
  }
  auto& dim = dim_entry->second;
  // Find value
  auto val_entry = dim.find(value);
  if (val_entry == dim.end()) {
    dim.insert({value, true});
  } else if (!val_entry->second) {
    val_entry->second = true;
  }
}

bool VariantBlacklist::contains(const std::string& dimension,
                                const std::string& value) {
  // Find dimension
  auto dim_entry = values.find(dimension);
  if (dim_entry != values.end()) {
    auto& dim = dim_entry->second;
    // Find value
    auto val_entry = dim.find(value);
    if (val_entry != dim.end() && val_entry->second == true) {
      return true;
    }
  }
  return false;
}

bool VariantBlacklist::is_blacklisted(const Variant& variant) {
  for (const auto& dimension : variant) {
    if (contains(dimension.first, dimension.second)) {
      return true;
    }
  }
  return false;
}

void VariantConfigurator::operator()(const Variant& variant) {
  for (const auto& option : variant) {
    std::cout << "[VariantConfigurator] Setting option: " << option.first
              << " = " << option.second << std::endl;
    VariableManager::instance().setVariableValue(option.first, option.second);
  }
}

}  // end namespace CoGaDB
